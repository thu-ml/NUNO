from timeit import default_timer
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.kdtree.tree import KDTree
from util.utilities import *
from .interp_mwno import MWT3d, get_initializer
from scipy.interpolate import LinearNDInterpolator, \
    NearestNDInterpolator, RegularGridInterpolator


################################################################
# Configs
################################################################
# Data path
PATH = 'data/heatsink/'
# Point cloud locations
PATH_XYZ = PATH + 'Heatsink_Output_XYZ.npy'
# Point cloud values (T, u, v, w, p)
PATH_U = PATH + 'Heatsink_Output_Function.npy'

# Dataset params
n_train = 900
n_test = 100
n_total = n_train + n_test
# The number of points in (output) point cloud
n_points = 19517

# MWNO configs
alpha = 8
c = 3
k = 3
nCZ = 4
L = 0

# Training params
batch_size = 20
learning_rate = 0.001
epochs = 501
patience = epochs // 20     # scheduler
reg_lambda = 5e-3

# Grid params
oversamp_ratio = 1.0        # used to calculate grid sizes
input_dim = 3               # (u, v, w)
output_dim = 1              # (T)

# K-D tree params
n_subdomains = 16


################################################################
# Training and evaluation
################################################################
def main(train_a, train_u, train_u_pc, test_a, test_u, test_u_pc):
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_a, train_u, 
            train_u_pc), 
        batch_size=batch_size, shuffle=True,
        generator=torch.Generator(device=device)
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_a, test_u, 
            test_u_pc),
        batch_size=batch_size, shuffle=False,
        generator=torch.Generator(device=device)
    )

    model = MWT3d(ich=input_dim*n_subdomains, 
            alpha = alpha,
            c = c,
            k = k, 
            base = 'legendre', # chebyshev
            nCZ = nCZ,
            L = L, och=output_dim*n_subdomains, 
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
        train_l2 = 0
        train_tot_loss = 0.0
        for x, y, y_pc in train_loader:
            optimizer.zero_grad()
            out = model(x).reshape(batch_size, 
                grid_shape[1], grid_shape[0], 
                grid_shape[2], output_dim, n_subdomains)
            out = y_normalizer.decode(out)
            loss1 = myloss(out, y)

            # Interpolation (from grids to point cloud)
            out = out.permute(0, 5, 4, 1, 2, 3)\
                .reshape(-1, output_dim, 
                    grid_shape[1], grid_shape[0], grid_shape[2])
                # Output shape: (batch * n_subdomains, output_dim
                #   s2, s1, s3)
            u = F.grid_sample(input=out, grid=xyz_sd, 
                padding_mode='border', align_corners=False)
                # Output shape: (batch * n_subdomains, output_dim, 
                #   n_points_sd_padded, 1, 1)
            out = u.squeeze(-1).squeeze(-1).permute(0, 2, 1)\
                .reshape(batch_size, n_subdomains, -1, output_dim)\
                .permute(0, 2, 3, 1)
                # Output shape: (batch_size, n_points_sd_padded, 
                #   output_dim, n_subdomains)
            
            out = out * input_u_sd_mask
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
                out = model(x).reshape(batch_size, 
                    grid_shape[1], grid_shape[0], 
                    grid_shape[2], output_dim, n_subdomains)
                out = y_normalizer.decode(out)

                test_l2 += myloss(out, y).item()

        train_l2 /= n_train
        test_l2 /= n_test

        t2 = default_timer()
        print("[Epoch {}] Time: {:.1f}s L2: {:>4e} Test_L2: {:>4e}"
                    .format(ep, t2-t1, train_l2, test_l2))

    with torch.no_grad():
        out = model(test_a_sd_grid).reshape(n_test, 
                grid_shape[1], grid_shape[0], 
                grid_shape[2], output_dim, n_subdomains)
        out = y_normalizer.decode(out).cpu().numpy()
        out = np.transpose(out, (0, 2, 1, 3, 4, 5))

        pred = np.zeros((n_test, max_n_points_sd, 
            output_dim, n_subdomains))
        for i in range(n_subdomains):
            bbox = bbox_sd[i]
            back_order = np.argsort(order_sd[i])
            _grid_shape = grid_shape[back_order]
            data = np.transpose(out[..., i], 
                (back_order+1).tolist() + [4, 0])

            grid_x = np.linspace(bbox[0][0], bbox[0][1], 
                num=_grid_shape[0])
            grid_y = np.linspace(bbox[1][0], bbox[1][1], 
                num=_grid_shape[1])
            grid_z = np.linspace(bbox[2][0], bbox[2][1], 
                num=_grid_shape[2])
            interp = RegularGridInterpolator(
                (grid_x, grid_y, grid_z), data)
            data = interp(xyz[indices_sd[i], :])
            data = np.transpose(data, (2, 0, 1))
            pred[:, :len(indices_sd[i]), :, i] = data

        pred = torch.tensor(pred).cpu()
        truth = test_u_point_cloud
        
        test_T_l2 = myloss(pred, truth).item()

    # Return final results
    return train_l2, test_l2, t2-t0, \
        test_T_l2 / n_test


if __name__ == "__main__":
    ################################################################
    # Load data and preprocessing
    ################################################################
    xyz = np.load(PATH_XYZ)                 # shape: (19517, 3)
    input_point_cloud = np.load(PATH_U)     # shape: (1000, 19517, 5)
    input_point_cloud = input_point_cloud[:n_total]

    print("Start KD-Tree splitting...")
    t1 = default_timer()
    point_cloud = xyz.tolist()
    # Use kd-tree to generate subdomain division
    tree= KDTree(
        point_cloud, dim=3, n_subdomains=n_subdomains, 
        n_blocks=8, return_indices=True
    )
    tree.solve()
    # Gather subdomain info
    bbox_sd = tree.get_subdomain_bounding_boxes()
    indices_sd = tree.get_subdomain_indices()
    # Pad the point cloud of each subdomain to the same size
    max_n_points_sd = np.max([len(indices_sd[i]) 
        for i in range(n_subdomains)])
    xyz_sd = np.zeros((1, max_n_points_sd, n_subdomains, 3))
    input_point_cloud_sd = np.zeros((n_total, 
        max_n_points_sd, input_point_cloud.shape[-1], n_subdomains))
    # Mask is used to ignore padded zeros when calculating errors
    input_u_sd_mask = np.zeros((1, max_n_points_sd, 1, n_subdomains))
    # The maximum grid shape of subdomains
    grid_shape = [-1] * 3
        # (s1, s2, s3)
    # The new coordinate order of each subdomain
    # (after long side alignment)
    order_sd = []
    for i in range(n_subdomains):
        # Normalize to [-1, 1]
        _xyz = xyz[indices_sd[i], :]
        _min, _max = np.min(_xyz, axis=0, keepdims=True), \
            np.max(_xyz, axis=0, keepdims=True)
        _xyz = (_xyz - _min) / (_max - _min) * 2 - 1
        # Long side alignment
        bbox = bbox_sd[i]
        scales = [bbox[j][1] - bbox[j][0] for j in range(3)]
        order = np.argsort(scales)
        _xyz = _xyz[:, order]
        order_sd.append(order.tolist())
        # Calculate the grid shape
        _grid_shape = cal_grid_shape(
            oversamp_ratio * len(indices_sd[i]), scales)
        _grid_shape.sort()
        grid_shape = np.maximum(grid_shape, _grid_shape)
        # Applying
        xyz_sd[0, :len(indices_sd[i]), i, :] = _xyz
        input_point_cloud_sd[:, :len(indices_sd[i]), :, i] = \
            input_point_cloud[:, indices_sd[i], :]
        input_u_sd_mask[0, :len(indices_sd[i]), 0, i] = 1.
    print(grid_shape)
    grid_shape = np.array(grid_shape)
    t2 = default_timer()
    print("Finish KD-Tree splitting, time elapsed: {:.1f}s".format(t2-t1))

    # Interpolation from point cloud to uniform grid
    t1 = default_timer()
    print("Start interpolation...")
    input_sd_grid = []
    point_cloud = xyz
    point_cloud_val = np.transpose(input_point_cloud, (1, 2, 0)) 
    interp_linear = LinearNDInterpolator(point_cloud, point_cloud_val)
    interp_nearest = NearestNDInterpolator(point_cloud, point_cloud_val)
    for i in range(n_subdomains):
        bbox = bbox_sd[i]
        _grid_shape = grid_shape[np.argsort(order_sd[i])]
        # Linear interpolation
        grid_x = np.linspace(bbox[0][0], bbox[0][1], 
            num=_grid_shape[0])
        grid_y = np.linspace(bbox[1][0], bbox[1][1], 
            num=_grid_shape[1])
        grid_z = np.linspace(bbox[2][0], bbox[2][1], 
            num=_grid_shape[2])
        grid_x, grid_y, grid_z = np.meshgrid(
            grid_x, grid_y, grid_z, indexing='ij')
        grid_val = interp_linear(grid_x, grid_y, grid_z)
        # Fill nan values
        nan_indices = np.isnan(grid_val)[..., 0, 0]
        fill_vals = interp_nearest(
            np.stack((
                grid_x[nan_indices], grid_y[nan_indices],
                grid_z[nan_indices]), axis=1))
        grid_val[nan_indices] = fill_vals
        # Long size alignment
        grid_val = np.transpose(grid_val, 
            order_sd[i] + [3, 4])
        input_sd_grid.append(np.transpose(grid_val, (4, 0, 1, 2, 3)))
    # Convert indexing to 'xy'
    input_sd_grid = np.transpose(
        np.array(input_sd_grid), (1, 3, 2, 4, 5, 0))

    t2 = default_timer()
    print("Finish interpolation, time elapsed: {:.1f}s".format(t2-t1))

    xyz_sd = torch.from_numpy(xyz_sd).cuda().float()
    xyz_sd = xyz_sd.repeat([batch_size, 1, 1, 1])\
        .permute(0, 2, 1, 3)\
        .reshape(batch_size * n_subdomains, -1, 1, 1, 3)
        # shape: (batch * n_subdomains, n_points_sd_padded, 1, 1, 3)
    input_point_cloud_sd = torch.from_numpy(input_point_cloud_sd).float()
        # shape: (ntotal, n_points_sd_padded, output_dim, n_subdomains)
    input_u_sd_mask = torch.from_numpy(input_u_sd_mask).cuda().float()
        # shape: (1, n_points_sd_padded, 1, n_subdomains)
    input_sd_grid = torch.from_numpy(input_sd_grid).float()
        # shape: (n_total, s2, s1, s3, input_dim + output_dim, n_subdomains)

    train_a_sd_grid = input_sd_grid[:n_train, ..., 1:4, :].\
        reshape(n_train, grid_shape[1], 
            grid_shape[0], grid_shape[2], -1).cuda()
    test_a_sd_grid = input_sd_grid[-n_test:, ..., 1:4, :].\
        reshape(n_test, grid_shape[1], 
            grid_shape[0], grid_shape[2], -1).cuda()

    input_sd_grid = input_sd_grid[..., 0:1, :]
    train_u_sd_grid = input_sd_grid[:n_train].cuda()
    test_u_sd_grid = input_sd_grid[-n_test:].cuda()

    input_point_cloud_sd = input_point_cloud_sd[..., 0:1, :]
    train_u_point_cloud = input_point_cloud_sd[:n_train].cuda()
    test_u_point_cloud = input_point_cloud_sd[-n_test:]

    a_normalizer = UnitGaussianNormalizer(train_a_sd_grid)
    train_a_sd_grid = a_normalizer.encode(train_a_sd_grid)
    test_a_sd_grid = a_normalizer.encode(test_a_sd_grid)

    y_normalizer = UnitGaussianNormalizer(train_u_sd_grid)

    ################################################################
    # Re-experiment with different random seeds
    ################################################################
    train_l2_res = []
    test_l2_res = []
    time_res = []
    test_T_l2_res = []
    for i in range(5):
        print("=== Round %d ==="%(i+1))
        set_random_seed(SEED_LIST[i])
        train_l2, test_l2, time, test_T_l2 = \
            main(train_a_sd_grid, train_u_sd_grid, 
            train_u_point_cloud, test_a_sd_grid, 
            test_u_sd_grid, test_u_point_cloud)
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
