"""
Reference
----------
author:   gaurav71531
source:   https://github.com/gaurav71531/mwt-operator
reminder: slightly modified, e.g., file path, better output format, etc.
"""
from timeit import default_timer
import math
import torch.nn.functional as F
from util.utilities import *
from util.util_mwno import MWT_CZ3d, get_initializer
from scipy.interpolate import LinearNDInterpolator, RBFInterpolator

    
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
PATH_XY = 'data/channel/Channel_Flow_XY.npy'
PATH_U = 'data/channel/Channel_Flow_Velocity_Pressure.npy'
PATH_U_G = 'data/channel/Preprocess_Channel_Flow_Velocity_Pressure_Grid.npy'

ntrain = 1000
ntest = 200
ntotal = ntrain + ntest
n_points = 3809

alpha = 8
c = 3
k = 3
nCZ = 4
L = 0

batch_size = 20
learning_rate = 0.001
epochs = 501
patience = epochs // 20

oversamp_ratio = 1.5
S = int(np.round(np.sqrt(oversamp_ratio * n_points)))
T_in = 15   # input: [0, 0.15)
T = 30      # output: [0.15, 0.30)
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

    model = MWT3d(ich=output_dim, 
            alpha = alpha,
            c = c,
            k = k, 
            base = 'legendre', # chebyshev
            nCZ = nCZ,
            L = L, och=output_dim, 
            initializer = get_initializer('xavier_normal')).cuda()
    print(count_params(model))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience)

    myloss = MultiLpLoss(size_average=False)
    y_normalizer.cuda()
    t0 = default_timer()
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_l2 = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            out = model(x).reshape(batch_size, S, S, -1)
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
            for x, y in test_loader:
                out = model(x).reshape(batch_size, S, S, -1)
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

