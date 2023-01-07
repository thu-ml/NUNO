"""
Reference
----------
author:   Zongyi Li
source:   https://github.com/neural-operator/fourier_neural_operator
reminder: slightly modified, e.g., file path, better output format, etc.
"""

from timeit import default_timer
import torch.nn.functional as F
from src.kdtree.tree import KDTree
from util.utilities import *
from scipy.interpolate import LinearNDInterpolator, RBFInterpolator

set_random_seed(SEED_LIST[0])

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
    def __init__(self, modes1, modes2, width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 15 timesteps + 2 locations (x_velocity, y_velocity, pressure, x, y)
        input shape: (batchsize, x=64, y=64, c=47 * 8)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=3 * 8)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 8 # pad the domain if input is non-periodic

        self.p = nn.Linear(45 * 8 + 2, self.width) # input channel is 47: the solution of the previous 15 timesteps + 2 locations
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
        self.q = MLP(self.width, 3 * 8, self.width * 4) # output channel is 3: (x_velocity, y_velocity, pressure)

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
PATH_U_SD = 'data/channel/Preprocess_Channel_Flow_Velocity_Pressure_Subdomain.npy'
PATH_U_SD_M = 'data/channel/Preprocess_Channel_Flow_Velocity_Pressure_Subdomain_Mask.npy'
PATH_U_SD_G = 'data/channel/Preprocess_Channel_Flow_Velocity_Pressure_Subdomain_Grid.npy'

ntrain = 1000
ntest = 200
ntotal = ntrain + ntest
n_points = 3809

modes = 12
width = 20

batch_size = 20
learning_rate = 0.001
epochs = 500
iterations = epochs*(ntrain//batch_size)

S = 24      # grid size: 24 x 24 x n_subdomains
T_in = 15   # input: [0, 0.15)
T = 30      # output: [0.15, 0.30)
step = 1

n_subdomains = 8
# Oversampling ratio (>=1) for preprocessing interpolation
oversamp_r1 = oversamp_r2 = 2

# Wether to save or load preprocessing results
SAVE_PREP = False
LOAD_PREP = True

################################################################
# load data and preprocessing
################################################################
input_xy = np.load(PATH_XY)            # shape: (3809, 2)
input_u = np.load(PATH_U)              # shape: (1200, 3809, 31, 3)

print("Start KD-Tree splitting...")
t1 = default_timer()
point_cloud = input_xy.tolist()
# Use kd-tree to generate subdomain dividing
tree= KDTree(
    point_cloud, dim=2, n_subdomains=n_subdomains, 
    n_blocks=8, return_indices=True
)
tree.solve()
# Gather subdomain info:
bbox_sd = tree.get_subdomain_bounding_boxes()
indices_sd = tree.get_subdomain_indices()
input_xy_sd = np.zeros((np.max([len(indices_sd[i]) for i in range(n_subdomains)]), n_subdomains, 2))
for i in range(n_subdomains):
    input_xy_sd[:len(indices_sd[i]), i, :] = input_xy[indices_sd[i], :]
t2 = default_timer()
print("Finish KD-Tree splitting, time elapsed: {:.1f}s".format(t2-t1))

if LOAD_PREP:
    input_u_sd_grid = np.load(PATH_U_SD_G)   # shape: (1200, 24, 24, 31, 3, n_subdomains) 
    input_u_sd = np.load(PATH_U_SD)          # shape: (1200, n_points_sd_padded, 31, 3, n_subdomains) 
    input_u_sd_mask = np.load(PATH_U_SD_M)   # shape: (1, n_points_sd_padded, 1, 1, n_subdomains) 
else:
    t1 = default_timer()
    print("Start interpolation...")
    # Interpolation from point cloud to uniform grid
    input_u_sd_grid = []
    point_cloud = input_xy
    point_cloud_val = np.transpose(input_u, (1, 2, 3, 0)) 
    interp_linear = LinearNDInterpolator(point_cloud, point_cloud_val)
    interp_rbf = RBFInterpolator(point_cloud, point_cloud_val, neighbors=6)
    for i in range(n_subdomains):
        # Uniform Grid
        n_points = len(indices_sd[i])
        bbox = bbox_sd[i]
        # Calculate the grid size, where the aspect ratio of the discrete grid 
        # remains the same as the that of the original subdomain (bbox)
        grid_size_x = np.sqrt(n_points * oversamp_r1 * \
            (bbox[0][1] - bbox[0][0]) / (bbox[1][1] - bbox[1][0]))
        grid_size_y = grid_size_x * (bbox[1][1] - bbox[1][0]) / (bbox[0][1] - bbox[0][0])
        grid_size_x, grid_size_y = max(int(np.round(grid_size_x)), 2), \
            max(int(np.round(grid_size_y)), 2)
        # Naive NUFFT
        grid_x = np.linspace(bbox_sd[i][0][0], bbox_sd[i][0][1], num=grid_size_x)
        grid_y = np.linspace(bbox_sd[i][1][0], bbox_sd[i][1][1], num=grid_size_y)
        grid_x, grid_y = np.meshgrid(grid_x, grid_y)
        grid_val = interp_linear(grid_x, grid_y)
        # Fill nan values
        nan_indices = np.isnan(grid_val)[..., 0, 0, 0]
        fill_vals = interp_rbf(np.stack((grid_x[nan_indices], grid_y[nan_indices]), axis=1))
        grid_val[nan_indices] = fill_vals
        freq = np.fft.rfft2(grid_val, axes=(0, 1))
        # Compute a square embeddings
        square_freq = np.zeros((S, S // 2 + 1, T+1, 3, ntotal)) + 0j
        square_freq[:min(S//2, freq.shape[0]//2), :min(S//2+1, freq.shape[1]//2+1), ...] = \
            freq[:min(S//2, freq.shape[0]//2), :min(S//2+1, freq.shape[1]//2+1), ...]
        square_freq[-min(S//2, freq.shape[0]//2):, :min(S//2+1, freq.shape[1]//2+1), ...] = \
            freq[-min(S//2, freq.shape[0]//2):, :min(S//2+1, freq.shape[1]//2+1), ...]
        grid_val = np.fft.irfft2(square_freq, s=(S, S), axes=(0, 1))
        input_u_sd_grid.append(np.transpose(grid_val, (4, 0, 1, 2, 3)))
    input_u_sd_grid = np.transpose(np.array(input_u_sd_grid), (1, 2, 3, 4, 5, 0))

    input_u_sd = np.zeros((ntotal, 
        np.max([len(indices_sd[i]) for i in range(n_subdomains)]), T+1, 3, n_subdomains))
    input_u_sd_mask = np.zeros((1, 
        np.max([len(indices_sd[i]) for i in range(n_subdomains)]), 1, 1, n_subdomains))
    for i in range(n_subdomains):
        input_u_sd[:, :len(indices_sd[i]), ..., i] = input_u[:, indices_sd[i], ...]
        input_u_sd_mask[:, :len(indices_sd[i]), ..., i] = 1.

    if SAVE_PREP:
        np.save(PATH_U_SD_G, input_u_sd_grid)
        np.save(PATH_U_SD, input_u_sd) 
        np.save(PATH_U_SD_M, input_u_sd_mask)
    t2 = default_timer()
    print("Finish interpolation, time elapsed: {:.1f}s".format(t2-t1))

input_xy_sd = torch.from_numpy(input_xy_sd).cuda().float()
input_xy_sd = input_xy_sd.unsqueeze(0).repeat([batch_size, 1, 1, 1])\
    .permute(0, 2, 1, 3)\
    .reshape(batch_size * n_subdomains, -1, 2)
    # shape: (batch * n_subdomains, n_points_sd_padded, 2)

input_u_sd_grid = torch.from_numpy(
    input_u_sd_grid.reshape(ntotal, S, S, T+1, -1)).float()
input_u_sd = torch.from_numpy(input_u_sd).float()
input_u_sd_mask = torch.from_numpy(input_u_sd_mask).cuda().float()

train_a_sd = input_u_sd_grid[:ntrain, ..., :T_in, :].cuda()
test_a_sd = input_u_sd_grid[-ntest:, ..., :T_in, :].cuda()

train_u_sd = input_u_sd[:ntrain, ..., T_in:T, :, :].cuda()
test_u_sd = input_u_sd[-ntest:, ..., T_in:T, :, :].cuda()

a_normalizer = UnitGaussianNormalizer(train_a_sd)
train_a_sd = a_normalizer.encode(train_a_sd)
test_a_sd = a_normalizer.encode(test_a_sd)

y_normalizer = UnitGaussianNormalizer(train_u_sd)

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(train_a_sd, train_u_sd), 
    batch_size=batch_size, shuffle=True,
    generator=torch.Generator(device=device)
)
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(test_a_sd, test_u_sd),
    batch_size=batch_size, shuffle=False,
    generator=torch.Generator(device=device)
)

################################################################
# training and evaluation
################################################################
model = FNO2d(modes, modes, width).cuda()
print(count_params(model))
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

myloss = LpLoss(size_average=False)
y_normalizer.cuda()
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    for xx, yy in train_loader:
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
        out = pred.reshape(batch_size, S, S, (T-T_in), 3, n_subdomains)\
            .permute(0, 5, 1, 2, 3, 4)\
            .reshape(-1, S, S, (T-T_in) * 3)
            # Output shape: (batch * n_subdomains, S, S, (T-T_in) * 3)

        # Interpolation (from grids to point cloud)
        # Normalize to [-1, 1]
        xy = input_xy_sd[...]
        _min, _max = torch.min(xy, dim=1, keepdim=True)[0], torch.max(xy, dim=1, keepdim=True)[0]
        xy = (xy - _min) / (_max - _min) * 2 - 1
        xy = xy.unsqueeze(-2)
            # Output shape: (batch * n_subdomains, n_points_sd_padded, 1, 2)
        u = F.grid_sample(input=out.permute(0, 3, 1, 2), grid=xy, 
            padding_mode='border', align_corners=False)
            # Output shape: (batch * n_subdomains, (T-T_in) * 3, n_points_sd_padded, 1)
        out = u.squeeze(-1).permute(0, 2, 1)\
            .reshape(batch_size, n_subdomains, -1, T-T_in, 3)\
            .permute(0, 2, 3, 4, 1)
            # Output shape: (batch_size, n_points_sd_padded, T-T_in, 3, n_subdomains)
        out = out * input_u_sd_mask

        out = y_normalizer.decode(out)
        l2 = myloss(out.reshape(batch_size, -1), yy.reshape(batch_size, -1))
        l2.backward()

        optimizer.step()
        scheduler.step()
        train_l2 += l2.item()

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for xx, yy in test_loader:
            # Time step marching 
            for t in range(0, T-T_in, step):
                im = model(xx.reshape(batch_size, S, S, -1))
                im = im.unsqueeze(-2)

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -2)
                
                xx = torch.cat((xx[..., step:, :], im), dim=-2)

            out = pred.reshape(batch_size, S, S, (T-T_in), 3, n_subdomains)\
                .permute(0, 5, 1, 2, 3, 4)\
                .reshape(-1, S, S, (T-T_in) * 3)
                # Output shape: (batch * n_subdomains, S, S, (T-T_in) * 3)

            # Interpolation (from grids to point cloud)
            # Normalize to [-1, 1]
            xy = input_xy_sd[...]
            _min, _max = torch.min(xy, dim=1, keepdim=True)[0], torch.max(xy, dim=1, keepdim=True)[0]
            xy = (xy - _min) / (_max - _min) * 2 - 1
            xy = xy.unsqueeze(-2)
                # Output shape: (batch * n_subdomains, n_points_sd_padded, 1, 2)
            u = F.grid_sample(input=out.permute(0, 3, 1, 2), grid=xy, 
                padding_mode='border', align_corners=False)
                # Output shape: (batch * n_subdomains, (T-T_in) * 3, n_points_sd_padded, 1)
            out = u.squeeze(-1).permute(0, 2, 1)\
                .reshape(batch_size, n_subdomains, -1, T-T_in, 3)\
                .permute(0, 2, 3, 4, 1)
                # Output shape: (batch_size, n_points_sd_padded, T-T_in, 3, n_subdomains)
            out = out * input_u_sd_mask

            out = y_normalizer.decode(out)
            test_l2 += myloss(out.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()

    train_l2 /= ntrain
    test_l2 /= ntest

    t2 = default_timer()
    print("[Epoch {}] Time: {:.1f}s L2: {:>4e} Test_L2: {:>4e}"
                .format(ep, t2-t1, train_l2, test_l2))
