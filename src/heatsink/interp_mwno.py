from timeit import default_timer
from tqdm import tqdm
import math
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from util.utilities import *
from util.util_mwno import MWT_CZ3d, get_initializer
from scipy.interpolate import LinearNDInterpolator, \
    NearestNDInterpolator, RegularGridInterpolator


class MWT3d(nn.Module):
    def __init__(self,
                ich = 1, k = 3, alpha = 2, c = 1,
                nCZ = 3,
                L = 0,
                base = 'legendre',
                initializer = None, och = 1,
                **kwargs):
        super(MWT3d,self).__init__()
        
        self.k = k
        self.c = c
        self.L = L
        self.nCZ = nCZ
        self.Lk = nn.Linear(ich, c*k**2)
        
        self.MWT_CZ = nn.ModuleList(
            [MWT_CZ3d(k, alpha, L, c, base, 
            initializer) for _ in range(nCZ)]
        )
        self.Lc0 = nn.Linear(c*k**2, 128)
        self.Lc1 = nn.Linear(128, och)
        
        if initializer is not None:
            self.reset_parameters(initializer)
        
    def forward(self, x):
        
        B, Nx, Ny, T, ich = x.shape # (B, Nx, Ny, T, d)
        ns = math.floor(np.log2(Nx))
        x = self.Lk(x)
        x = x.view(B, Nx, Ny, T, self.c, self.k**2)
    
        for i in range(self.nCZ):
            x = self.MWT_CZ[i](x)
            if i < self.nCZ-1:
                x = F.relu(x)

        # De-padding
        x = x[:, :Nx, :Ny, ...]
        x = x.view(B, Nx, Ny, T, -1) # collapse c and k**2
        x = self.Lc0(x)
        x = F.relu(x)
        x = self.Lc1(x)
        return x
    
    def reset_parameters(self, initializer):
        initializer(self.Lc0.weight)
        initializer(self.Lc1.weight)


################################################################
# configs
################################################################
PATH_XYZ = 'data/heatsink/Heatsink_Output_XYZ.npy'
PATH_U = 'data/heatsink/Heatsink_Output_Function.npy'

ntrain = 900
ntest = 100
ntotal = ntrain + ntest
n_points = 19517

alpha = 8
c = 3
k = 3
nCZ = 4
L = 0

batch_size = 20
learning_rate = 0.001
epochs = 501
patience = epochs // 20
reg_lambda = 5e-3

oversamp_ratio = 1.0
grid_size = oversamp_ratio * n_points
input_dim = 3
output_dim = 1

################################################################
# training and evaluation
################################################################
def main(train_a, train_u, train_u_pc, test_a, test_u, test_u_pc):
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_a, train_u, train_u_pc), 
        batch_size=batch_size, shuffle=True,
        generator=torch.Generator(device=device)
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_a, test_u, test_u_pc), 
        batch_size=batch_size, shuffle=False,
        generator=torch.Generator(device=device)
    )

    model = MWT3d(ich=input_dim, 
            alpha = alpha,
            c = c,
            k = k, 
            base = 'legendre', # chebyshev
            nCZ = nCZ,
            L = L, och=output_dim, 
            initializer = get_initializer('xavier_normal')).cuda()
    print(count_params(model))
    optimizer = Adam(model.parameters(), 
        lr=learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=patience)

    myloss = LpLoss(size_average=False)
    regloss = nn.MSELoss()
    y_normalizer.cuda()
    t0 = default_timer()
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_l2 = 0.0
        train_tot_loss = 0.0
        for x, y, y_pc in train_loader:
            optimizer.zero_grad()
            out = model(x)
                # Output shape: (batch, s2, s1, s3, output_dim)
            out = y_normalizer.decode(out)
            loss1 = myloss(out, y)

            # Interpolation (from grids to point cloud)
            # Normalize to [-1, 1]
            xyz = input_xyz[...]
            _min, _max = torch.min(xyz, dim=1, keepdim=True)[0], torch.max(xyz, dim=1, keepdim=True)[0]
            xyz = (xyz - _min) / (_max - _min) * 2 - 1
            xyz = xyz.unsqueeze(-2).unsqueeze(-2)
                # Output shape: (batch, n_points, 1, 1, 3)
            u = F.grid_sample(input=out.permute(0, 4, 1, 2, 3), grid=xyz, 
                padding_mode='border', align_corners=False)
                # Output shape: (batch, output_dim, n_points, 1, 1)
            out = u.squeeze(-1).squeeze(-1).permute(0, 2, 1)
                # Output shape: (batch_size, n_points, output_dim)

            loss = loss1 + reg_lambda * regloss(out, y_pc)
            loss.backward()

            optimizer.step()
            train_l2 += loss1.item()
            train_tot_loss += loss.item()

        scheduler.step(train_tot_loss)

        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y, _ in test_loader:
                out = model(x)
                    # Output shape: (batch, s2, s1, s3, output_dim)
                out = y_normalizer.decode(out)

                test_l2 += myloss(out, y).item()

        train_l2 /= ntrain
        test_l2 /= ntest

        t2 = default_timer()
        print("[Epoch {}] Time: {:.1f}s L2: {:>4e} Test_L2: {:>4e}"
                    .format(ep, t2-t1, train_l2, test_l2))
    
    with torch.no_grad():
        pred = model(test_a_grid)
        pred = y_normalizer.decode(pred).cpu().numpy()
        pred = np.transpose(pred, (2, 1, 3, 4, 0))
        grid_x = np.linspace(np.min(point_cloud[:, 0]), 
            np.max(point_cloud[:, 0]), num=grid_shape[0])
        grid_y = np.linspace(np.min(point_cloud[:, 1]), 
            np.max(point_cloud[:, 1]), num=grid_shape[1])
        grid_z = np.linspace(np.min(point_cloud[:, 2]), 
            np.max(point_cloud[:, 2]), num=grid_shape[2])
        interp = RegularGridInterpolator(
            (grid_x, grid_y, grid_z), pred)
        pred = interp(point_cloud)
        pred = np.transpose(pred, (2, 0, 1))
        pred = torch.tensor(pred).cpu()
        truth = test_u_point_cloud
        
        test_T_l2 = myloss(pred, truth).item()

    # Return final results
    return train_l2, test_l2, t2-t0, \
        test_T_l2 / ntest


if __name__ == "__main__":
    ################################################################
    # load data and preprocessing
    ################################################################
    input_xyz = np.load(PATH_XYZ)           # shape: (19517, 3)
    input_point_cloud = np.load(PATH_U)     # shape: (1000, 19517, 5)
    input_point_cloud = input_point_cloud[:ntotal]
    # Calculate the grid shape
    scales = np.max(input_xyz, 0) - np.min(input_xyz, 0)
    grid_shape = cal_grid_shape(grid_size, scales)
        # (s1, s2, s3)
    print(grid_shape)
    # Interpolation from point cloud to uniform grid
    t1 = default_timer()
    print("Start interpolation...")
    point_cloud = input_xyz
    point_cloud_val = np.transpose(input_point_cloud, (1, 2, 0)) 
    interp_linear = LinearNDInterpolator(point_cloud, point_cloud_val)
    interp_nearest = NearestNDInterpolator(point_cloud, point_cloud_val)
    # Uniform Grid
    grid_x = np.linspace(np.min(point_cloud[:, 0]), 
        np.max(point_cloud[:, 0]), num=grid_shape[0])
    grid_y = np.linspace(np.min(point_cloud[:, 1]), 
        np.max(point_cloud[:, 1]), num=grid_shape[1])
    grid_z = np.linspace(np.min(point_cloud[:, 2]), 
        np.max(point_cloud[:, 2]), num=grid_shape[2])
    grid_x, grid_y, grid_z = np.meshgrid(grid_x, grid_y, grid_z)
    grid_val = interp_linear(grid_x, grid_y, grid_z)
    # Fill nan values
    nan_indices = np.isnan(grid_val)[..., 0, 0]
    fill_vals = interp_nearest(np.stack((
        grid_x[nan_indices], grid_y[nan_indices],
        grid_z[nan_indices]), axis=-1))
    grid_val[nan_indices] = fill_vals

    input_grid = np.transpose(grid_val, (4, 0, 1, 2, 3)) 
        # shape: (1000, s2, s1, s3, 5)
    t2 = default_timer()
    print("Finish interpolation, time elapsed: {:.1f}s".format(t2-t1))

    input_xyz = torch.from_numpy(input_xyz).cuda().float()
    input_xyz = input_xyz.unsqueeze(0).repeat([batch_size, 1, 1])

    input_grid = torch.from_numpy(input_grid).float()
    input_point_cloud = torch.from_numpy(input_point_cloud).float()

    train_a_grid = input_grid[:ntrain, ..., 1:4].cuda()
    test_a_grid = input_grid[-ntest:, ..., 1:4].cuda()

    input_grid = input_grid[..., 0:1]
    train_u_grid = input_grid[:ntrain].cuda()
    test_u_grid = input_grid[-ntest:].cuda()

    input_point_cloud = input_point_cloud[..., 0:1]
    train_u_point_cloud = input_point_cloud[:ntrain].cuda()
    test_u_point_cloud = input_point_cloud[-ntest:]

    a_normalizer = UnitGaussianNormalizer(train_a_grid)
    train_a_grid = a_normalizer.encode(train_a_grid)
    test_a_grid = a_normalizer.encode(test_a_grid)

    y_normalizer = UnitGaussianNormalizer(train_u_grid)

    ################################################################
    # re-experiment with different random seeds
    ################################################################
    train_l2_res = []
    test_l2_res = []
    time_res = []
    test_T_l2_res = []
    for i in range(5):
        print("=== Round %d ==="%(i+1))
        set_random_seed(SEED_LIST[i])
        train_l2, test_l2, time, test_T_l2 = \
            main(train_a_grid, train_u_grid, train_u_point_cloud, 
                test_a_grid, test_u_grid, test_u_point_cloud)
        train_l2_res.append(train_l2)
        test_l2_res.append(test_l2)
        time_res.append(time)
        test_T_l2_res.append(test_T_l2)
    print("=== Finish ===")
    for i in range(5):
        print('''[Round {}] Time: {:.1f}s Train_L2: {:>4e} Test_L2: {:>4e}
            \tT_L2: {:>4e}'''
            .format(i+1, time_res[i], train_l2_res[i], test_l2_res[i], 
            test_T_l2_res[i]))
