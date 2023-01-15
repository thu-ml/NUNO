"""
Reference
----------
author:   Zongyi Li
source:   https://github.com/neural-operator/fourier_neural_operator
reminder: slightly modified, e.g., file path, better output format, etc.
"""

from timeit import default_timer
from tqdm import tqdm
import torch.nn.functional as F
from util.utilities import *
from scipy.interpolate import LinearNDInterpolator, RBFInterpolator


################################################################
# fourier layer
################################################################

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, in_channels=47, out_channels=3):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 15 timesteps + 2 locations (x_velocity, y_velocity, pressure, x, y)
        input shape: (batchsize, x=64, y=64, c=47)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=3)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 8 # pad the domain if input is non-periodic

        self.p = nn.Linear(in_channels, self.width) # input channel is 47: the solution of the previous 15 timesteps + 2 locations
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.norm = nn.InstanceNorm2d(self.width)
        self.q = MLP(self.width, out_channels, self.width * 4) # output channel is 3: (x_velocity, y_velocity, pressure)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)
        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.norm(self.conv0(self.norm(x)))
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.norm(self.conv1(self.norm(x)))
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.norm(self.conv2(self.norm(x)))
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.norm(self.conv3(self.norm(x)))
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = self.q(x)
        x = x.permute(0, 2, 3, 1)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
################################################################
# configs
################################################################
PATH_XY = 'data/channel/Channel_Flow_XY.npy'
PATH_U = 'data/channel/Channel_Flow_Velocity_Pressure.npy'
PATH_U_G = 'data/channel/Preprocess_Channel_Flow_Velocity_Pressure_Grid.npy'

ntrain = 1000
ntest = 200
ntotal = ntrain + ntest
n_points = 3809

modes = 12
width = 20

batch_size = 20
learning_rate = 0.001
epochs = 501
patience = epochs // 20

S = 64      # grid size: 64 x 64
T_in = 15   # input: [0, 0.15)
T = 30      # output: [0.15, 0.30)
step = 1
output_dim = 3

# Wether to save or load preprocessing results
SAVE_PREP = False
LOAD_PREP = True

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

    model = FNO2d(modes, modes, width).cuda()
    print(count_params(model))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience)

    myloss = MultiLpLoss(size_average=False)
    y_normalizer.cuda()
    t0 = default_timer()
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_l2 = 0.0
        for xx, y in train_loader:
            # Time step marching
            for t in range(0, T-T_in, step):
                im = model(xx.reshape(batch_size, S, S, -1))
                im = im.unsqueeze(-2)

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -2)
                
                xx = torch.cat((xx[..., step:, :], im), dim=-2)

            optimizer.zero_grad()
            out = pred.reshape(batch_size, S, S, -1)
                # Output shape: (batch, S, S, (T-T_in) * 3)

            # Interpolation (from grids to point cloud)
            # Normalize to [-1, 1]
            xy = input_xy[...]
            _min, _max = torch.min(xy, dim=1, keepdim=True)[0], torch.max(xy, dim=1, keepdim=True)[0]
            xy = (xy - _min) / (_max - _min) * 2 - 1
            xy = xy.unsqueeze(-2)
                # Output shape: (batch, n_points, 1, 2)
            u = F.grid_sample(input=out.permute(0, 3, 1, 2), grid=xy, 
                padding_mode='border', align_corners=False)
                # Output shape: (batch, (T-T_in) * 3, n_points, 1)
            out = u.squeeze(-1).permute(0, 2, 1)\
                .reshape(batch_size, n_points, T-T_in, 3)
                # Output shape: (batch_size, n_points, T-T_in, 3)

            out = y_normalizer.decode(out)
            out = out.reshape(batch_size, n_points, T-T_in, output_dim)
            l2 = myloss(out, y.reshape_as(out))
            l2.backward()

            optimizer.step()
            train_l2 += l2.item()

        scheduler.step(train_l2)

        model.eval()
        test_l2 = 0.0
        test_u_l2 = 0.0
        test_v_l2 = 0.0
        test_p_l2 = 0.0
        with torch.no_grad():
            for xx, y in test_loader:
                # Time step marching 
                for t in range(0, T-T_in, step):
                    im = model(xx.reshape(batch_size, S, S, -1))
                    im = im.unsqueeze(-2)

                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -2)
                    
                    xx = torch.cat((xx[..., step:, :], im), dim=-2)

                out = pred.reshape(batch_size, S, S, -1)
                    # Output shape: (batch, S, S, (T-T_in) * 3)

                # Interpolation (from grids to point cloud)
                # Normalize to [-1, 1]
                xy = input_xy[...]
                _min, _max = torch.min(xy, dim=1, keepdim=True)[0], torch.max(xy, dim=1, keepdim=True)[0]
                xy = (xy - _min) / (_max - _min) * 2 - 1
                xy = xy.unsqueeze(-2)
                    # Output shape: (batch, n_points, 1, 2)
                u = F.grid_sample(input=out.permute(0, 3, 1, 2), grid=xy, 
                    padding_mode='border', align_corners=False)
                    # Output shape: (batch, (T-T_in) * 3, n_points, 1)
                out = u.squeeze(-1).permute(0, 2, 1)\
                    .reshape(batch_size, n_points, T-T_in, 3)
                    # Output shape: (batch_size, n_points, T-T_in, 3)

                out = y_normalizer.decode(out)
                out = out.reshape(batch_size, n_points, T-T_in, output_dim)
                y = y.reshape_as(out)
                test_u_l2 += myloss(out[..., 0], y[..., 0], multi_channel=False).item()
                test_v_l2 += myloss(out[..., 1], y[..., 1], multi_channel=False).item()
                test_p_l2 += myloss(out[..., 2], y[..., 2], multi_channel=False).item()

        train_l2 /= ntrain
        test_u_l2/=ntest
        test_v_l2/=ntest
        test_p_l2/=ntest
        test_l2 = (test_u_l2 + test_v_l2 + test_p_l2)/3

        t2 = default_timer()
        print("[Epoch {}] Time: {:.1f}s L2: {:>4e} Test_L2: {:>4e}"
                    .format(ep, t2-t1, train_l2, test_l2))
    
    # Return final results
    return train_l2, test_l2, t2-t0, test_u_l2, test_v_l2, test_p_l2


if __name__ == "__main__":
    ################################################################
    # load data and preprocessing
    ################################################################
    input_xy = np.load(PATH_XY)            # shape: (3809, 2)
    input_u = np.load(PATH_U)              # shape: (1200, 3809, 31, 3)
    if LOAD_PREP:
        input_u_grid = np.load(PATH_U_G)   # shape: (1200, 64, 64, 31, 3) 
    else:
        t1 = default_timer()
        print("Start interpolation...")
        # Interpolation from point cloud to uniform grid
        point_cloud = input_xy
        point_cloud_val = np.transpose(input_u, (1, 2, 3, 0)) 
        interp_linear = LinearNDInterpolator(point_cloud, point_cloud_val)
        interp_rbf = RBFInterpolator(point_cloud, point_cloud_val, neighbors=6)
        # Uniform Grid
        grid_x = np.linspace(np.min(point_cloud[:, 0]), np.max(point_cloud[:, 0]), num=S)
        grid_y = np.linspace(np.min(point_cloud[:, 1]), np.min(point_cloud[:, 1]), num=S)
        grid_x, grid_y = np.meshgrid(grid_x, grid_y)
        grid_val = interp_linear(grid_x, grid_y)
        # Fill nan values
        nan_indices = np.isnan(grid_val)[..., 0, 0, 0]
        fill_vals = interp_rbf(np.stack((grid_x[nan_indices], grid_y[nan_indices]), axis=1))
        grid_val[nan_indices] = fill_vals

        input_u_grid = np.transpose(grid_val, (4, 0, 1, 2, 3)) 
        if SAVE_PREP:
            np.save(PATH_U_G, input_u_grid)
        t2 = default_timer()
        print("Finish interpolation, time elapsed: {:.1f}s".format(t2-t1))

    input_xy = torch.from_numpy(input_xy).cuda().float()
    input_xy = input_xy.unsqueeze(0).repeat([batch_size, 1, 1])

    input_u_grid = torch.from_numpy(input_u_grid).float()
    input_u = torch.from_numpy(input_u).float()

    train_a = input_u_grid[:ntrain, ..., :T_in, :].cuda()
    test_a = input_u_grid[-ntest:, ..., :T_in, :].cuda()

    train_u = input_u[:ntrain, ..., T_in:T, :].cuda()
    test_u = input_u[-ntest:, ..., T_in:T, :].cuda()

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
    test_u_l2_res = []
    test_v_l2_res = []
    test_p_l2_res = []
    for i in range(5):
        print("=== Round %d ==="%(i+1))
        set_random_seed(SEED_LIST[i])
        train_l2, test_l2, time, test_u_l2, test_v_l2, test_p_l2 = \
            main(train_a, train_u, test_a, test_u)
        train_l2_res.append(train_l2)
        test_l2_res.append(test_l2)
        time_res.append(time)
        test_u_l2_res.append(test_u_l2)
        test_v_l2_res.append(test_v_l2)
        test_p_l2_res.append(test_p_l2)
    print("=== Finish ===")
    for i in range(5):
        print('''[Round {}] Time: {:.1f}s Train_L2: {:>4e} Test_L2: {:>4e}
            \tu_L2: {:>4e} v_L2: {:>4e} p_L2: {:>4e}'''
            .format(i+1, time_res[i], train_l2_res[i], test_l2_res[i], 
            test_u_l2_res[i], test_v_l2_res[i], test_p_l2_res[i]))

