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
    def __init__(self, modes1, modes2, modes3, width):
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

        self.p = nn.Linear(27, self.width) # input channel is 6: (x_velocity, y_velocity, pressure) * n_subdomains + 3 locations (u, v, p, x, y, t)
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
        self.q = MLP(self.width, 24, self.width * 4) # output channel is 24: (u, v, p) * n_subdomains

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
PATH_XY = 'data/channel/Channel_Flow_XY.npy'
PATH_U = 'data/channel/Channel_Flow_Velocity_Pressure.npy'
PATH_U_SD = 'data/channel/Preprocess_Channel_Flow_Velocity_Pressure_Subdomain.npy'
PATH_U_SD_M = 'data/channel/Preprocess_Channel_Flow_Velocity_Pressure_Subdomain_Mask.npy'
PATH_U_SD_G = 'data/channel/Preprocess_Channel_Flow_Velocity_Pressure_Subdomain_Grid.npy'

ntrain = 1000
ntest = 200
ntotal = ntrain + ntest
n_points = 3809

modes = 8
width = 20

batch_size = 10
learning_rate = 0.001
epochs = 500
iterations = epochs*(ntrain//batch_size)

S = 24      # grid size: 24 x 24 x n_subdomains
T_in = 15   # input: [0, 0.15)
T = 30      # output: [0.15, 0.30)

n_subdomains = 8

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
borders_sd = tree.get_subdomain_borders(shrink=True)
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
        grid_x = np.linspace(borders_sd[i][0][0], borders_sd[i][0][1], num=S)
        grid_y = np.linspace(borders_sd[i][1][0], borders_sd[i][1][1], num=S)
        grid_x, grid_y = np.meshgrid(grid_x, grid_y)
        grid_val = interp_linear(grid_x, grid_y)
        # Fill nan values
        nan_indices = np.isnan(grid_val)[..., 0, 0, 0]
        fill_vals = interp_rbf(np.stack((grid_x[nan_indices], grid_y[nan_indices]), axis=1))
        grid_val[nan_indices] = fill_vals
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

train_a_sd = input_u_sd_grid[:ntrain, ..., :T_in, :]
test_a_sd = input_u_sd_grid[-ntest:, ..., :T_in, :]

train_u_sd = input_u_sd[:ntrain, ..., T_in:T, :, :]
test_u_sd = input_u_sd[-ntest:, ..., T_in:T, :, :]

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
model = FNO3d(modes, modes, modes, width).cuda()
print(count_params(model))
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

myloss = LpLoss(size_average=False)
y_normalizer.cuda()
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        out = model(x).reshape(batch_size, S, S, (T-T_in), 3, n_subdomains)\
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
        l2 = myloss(out.reshape(batch_size, -1), y.reshape(batch_size, -1))
        l2.backward()

        optimizer.step()
        scheduler.step()
        train_l2 += l2.item()

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()

            out = model(x).reshape(batch_size, S, S, (T-T_in), 3, n_subdomains)\
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
            test_l2 += myloss(out.reshape(batch_size, -1), y.reshape(batch_size, -1)).item()

    train_l2 /= ntrain
    test_l2 /= ntest

    t2 = default_timer()
    print("[Epoch {}] Time: {:.1f}s L2: {:>4e} Test_L2: {:>4e}"
                .format(ep, t2-t1, train_l2, test_l2))
