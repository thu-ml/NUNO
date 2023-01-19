from timeit import default_timer
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from util.utilities import *
from scipy.interpolate import LinearNDInterpolator, RBFInterpolator
from .encoder import Encoder


################################################################
# 3d fourier layers
################################################################
class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv3d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv3d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class FNO3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width, in_channels=3, out_channels=3):
        super(FNO3d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: (x_velocity, y_velocity, pressure) in [0, T)
        input shape: (batchsize, x=64, y=64, t=T, c=3)
        output: (x_velocity, y_velocity, pressure) in [T, 2T)
        output shape: (batchsize, x=64, y=64, t=T, c=3)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.padding = 6 # pad the domain if input is non-periodic

        self.p = nn.Linear(in_channels+3, self.width) # input channel is 6: (x_velocity, y_velocity, pressure) + 3 locations (u, v, p, x, y, z)
        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)
        self.q = MLP(self.width, out_channels, self.width * 4) # output channel is 3: (u, v, p)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding]
        x = self.q(x)
        x = x.permute(0, 2, 3, 4, 1) # pad the domain if input is non-periodic
        return x


    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)

################################################################
# configs
################################################################
PATH_INP_XY = 'data/heatsink/Heatsink_Input_XY.npy'
PATH_INP_A = 'data/heatsink/Heatsink_Input_Function.npy'
PATH_OUP_XYZ = 'data/heatsink/Heatsink_Output_XYZ.npy'
PATH_OUP_U = 'data/heatsink/Heatsink_Output_Function.npy'

ntrain = 1000
ntest = 100
ntotal = ntrain + ntest
n_points = 19517

modes = 8
width = 20

batch_size = 20
learning_rate = 0.001
epochs = 501
patience = epochs // 20

oversamp_ratio = 1.0
grid_size = oversamp_ratio * n_points
input_dim = 1
output_dim = 5

################################################################
# training and evaluation
################################################################
def main(train_a, train_u, test_a, test_u):
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_a, train_u), 
        batch_size=batch_size, shuffle=True,
        generator=torch.Generator(device=device)
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_a, test_u), 
        batch_size=batch_size, shuffle=False,
        generator=torch.Generator(device=device)
    )

    model = FNO3d(modes, modes, modes, width, 
        in_channels=input_dim, out_channels=output_dim).cuda()
    model_encoder = Encoder(target_size=grid_shape[0]).cuda()
    print(count_params(model) + count_params(model_encoder))
    params = list(model.parameters()) + \
        list(model_encoder.parameters())
    optimizer = Adam(params, 
        lr=learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=patience)

    myloss = MultiLpLoss(size_average=False)
    y_normalizer.cuda()
    t0 = default_timer()
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_l2 = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            x = model_encoder(x)
            out = model(x)
                # Output shape: (batch, s3, s2, s1, output_dim)

            # Interpolation (from grids to point cloud)
            # Normalize to [-1, 1]
            xyz = output_xyz[...]
            _min, _max = torch.min(xyz, dim=1, keepdim=True)[0], torch.max(xyz, dim=1, keepdim=True)[0]
            xyz = (xyz - _min) / (_max - _min) * 2 - 1
            xyz = xyz.unsqueeze(-2).unsqueeze(-2)
                # Output shape: (batch, n_points, 1, 1, 3)
            u = F.grid_sample(input=out.permute(0, 4, 1, 2, 3), grid=xyz, 
                padding_mode='border', align_corners=False)
                # Output shape: (batch, output_dim, n_points, 1, 1)
            out = u.squeeze(-1).squeeze(-1).permute(0, 2, 1)
                # Output shape: (batch_size, n_points, output_dim)

            out = y_normalizer.decode(out)
            out = out * output_u_mask
            l2 = myloss(out, y)
            l2.backward()

            optimizer.step()
            train_l2 += l2.item()

        scheduler.step(train_l2)

        model.eval()
        test_l2 = 0.0
        test_T_l2 = 0.0
        test_u_l2 = 0.0
        test_v_l2 = 0.0
        test_w_l2 = 0.0
        test_p_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x = model_encoder(x)
                out = model(x)
                    # Output shape: (batch, s3, s2, s1, output_dim)

                # Interpolation (from grids to point cloud)
                # Normalize to [-1, 1]
                xyz = output_xyz[...]
                _min, _max = torch.min(xyz, dim=1, keepdim=True)[0], torch.max(xyz, dim=1, keepdim=True)[0]
                xyz = (xyz - _min) / (_max - _min) * 2 - 1
                xyz = xyz.unsqueeze(-2).unsqueeze(-2)
                    # Output shape: (batch, n_points, 1, 1, 3)
                u = F.grid_sample(input=out.permute(0, 4, 1, 2, 3), grid=xyz, 
                    padding_mode='border', align_corners=False)
                    # Output shape: (batch, output_dim, n_points, 1, 1)
                out = u.squeeze(-1).squeeze(-1).permute(0, 2, 1)
                    # Output shape: (batch_size, n_points, output_dim)

                out = y_normalizer.decode(out)
                out = out * output_u_mask
                test_T_l2 += myloss(out[..., 0], y[..., 0], multi_channel=False).item()
                test_u_l2 += myloss(out[..., 1], y[..., 1], multi_channel=False).item()
                test_v_l2 += myloss(out[..., 2], y[..., 2], multi_channel=False).item()
                test_w_l2 += myloss(out[..., 3], y[..., 3], multi_channel=False).item()
                test_p_l2 += myloss(out[..., 4], y[..., 4], multi_channel=False).item()

        train_l2 /= ntrain
        test_T_l2/=ntest
        test_u_l2/=ntest
        test_v_l2/=ntest
        test_w_l2/=ntest
        test_p_l2/=ntest
        test_l2 = (
            test_T_l2 + test_u_l2 +
            test_v_l2 + test_w_l2 + test_p_l2)/3

        t2 = default_timer()
        print("[Epoch {}] Time: {:.1f}s L2: {:>4e} Test_L2: {:>4e}"
                    .format(ep, t2-t1, train_l2, test_l2))
    
    # Return final results
    return train_l2, test_l2, t2-t0, \
        test_T_l2, test_u_l2, test_v_l2, test_w_l2, test_p_l2


if __name__ == "__main__":
    ################################################################
    # load data and preprocessing
    ################################################################
    input_xy = np.load(PATH_INP_XY)         # shape: (85, 2)
    input_a = np.load(PATH_INP_A)           # shape: (1100, 85, 1)
    output_xyz = np.load(PATH_OUP_XYZ)      # shape: (19517, 3)
    output_u = np.load(PATH_OUP_U)          # shape: (1100, 19517, 5)
    # Mask the point where u, v, w, p has no definition
    output_u_mask = np.where(output_u[0] == 0, 0, 1)
        # shape: (19517, 5)
    # Calculate the grid shape
    scales = np.max(output_xyz, 0) - np.min(output_xyz, 0)
    grid_shape = cal_grid_shape(grid_size, scales)
        # (s1, s2, s3)
    print(grid_shape)
    # Interpolation from point cloud to uniform grid
    t1 = default_timer()
    print("Start interpolation...")
    point_cloud = input_xy
    point_cloud_val = np.transpose(input_a, (1, 2, 0)) 
    interp_linear = LinearNDInterpolator(point_cloud, point_cloud_val)
    interp_rbf = RBFInterpolator(point_cloud, point_cloud_val, neighbors=6)
    # Uniform Grid
    grid_x = np.linspace(np.min(point_cloud[:, 0]), 
        np.max(point_cloud[:, 0]), num=grid_shape[1])
    grid_y = np.linspace(np.min(point_cloud[:, 1]), 
        np.max(point_cloud[:, 1]), num=grid_shape[2])
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)
    grid_val = interp_linear(grid_x, grid_y)
    # Fill nan values
    nan_indices = np.isnan(grid_val)[..., 0, 0]
    fill_vals = interp_rbf(np.stack((grid_x[nan_indices], grid_y[nan_indices]), axis=1))
    grid_val[nan_indices] = fill_vals

    input_a_grid = np.transpose(grid_val, (3, 0, 1, 2)) 
    input_a_grid = np.expand_dims(input_a_grid, axis=-1)
        # shape: (1100, s3, s2, 1, 1)
    t2 = default_timer()
    print("Finish interpolation, time elapsed: {:.1f}s".format(t2-t1))

    output_xyz = torch.from_numpy(output_xyz).cuda().float()
    output_xyz = output_xyz.unsqueeze(0).repeat([batch_size, 1, 1])
    output_u_mask = torch.from_numpy(output_u_mask).cuda().float()
    output_u_mask = output_u_mask.unsqueeze(0).repeat([batch_size, 1, 1])

    input_a_grid = torch.from_numpy(input_a_grid).float()
    output_u = torch.from_numpy(output_u).float()

    train_a = input_a_grid[:ntrain].cuda()
    test_a = input_a_grid[-ntest:].cuda()

    train_u = output_u[:ntrain].cuda()
    test_u = output_u[-ntest:].cuda()

    a_normalizer = UnitGaussianNormalizer(train_a)
    train_a = a_normalizer.encode(train_a)
    test_a = a_normalizer.encode(test_a)

    y_normalizer = UnitGaussianNormalizer(train_u)

    ################################################################
    # re-experiment with different random seeds
    ################################################################
    train_l2_res = []
    test_l2_res = []
    time_res = []
    test_T_l2_res = []
    test_u_l2_res = []
    test_v_l2_res = []
    test_w_l2_res = []
    test_p_l2_res = []
    for i in range(5):
        print("=== Round %d ==="%(i+1))
        set_random_seed(SEED_LIST[i])
        train_l2, test_l2, time, test_T_l2, \
        test_u_l2, test_v_l2, test_w_l2, test_p_l2 = \
            main(train_a, train_u, test_a, test_u)
        train_l2_res.append(train_l2)
        test_l2_res.append(test_l2)
        time_res.append(time)
        test_T_l2_res.append(test_T_l2)
        test_u_l2_res.append(test_u_l2)
        test_v_l2_res.append(test_v_l2)
        test_w_l2_res.append(test_w_l2)
        test_p_l2_res.append(test_p_l2)
    print("=== Finish ===")
    for i in range(5):
        print('''[Round {}] Time: {:.1f}s Train_L2: {:>4e} Test_L2: {:>4e}
            \tT_L2: {:>4e} u_L2: {:>4e} v_L2: {:>4e} w_L2: {:>4e} p_L2: {:>4e}'''
            .format(i+1, time_res[i], train_l2_res[i], test_l2_res[i], 
            test_T_l2_res[i], test_u_l2_res[i], test_v_l2_res[i], 
            test_w_l2_res[i], test_p_l2_res[i]))
