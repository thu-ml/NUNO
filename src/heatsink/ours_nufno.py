from timeit import default_timer
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.kdtree.tree import KDTree
from util.utilities import *
from .interp_fno import FNO3d
from .encoder import Encoder
from scipy.interpolate import LinearNDInterpolator, RBFInterpolator


################################################################
# Configs
################################################################
# Data path
PATH = 'data/heatsink/'
# Input point cloud locations 
PATH_INP_XY = PATH + 'Heatsink_Input_XY.npy'
# Input point cloud values (u0)
PATH_INP_A = PATH + 'Heatsink_Input_Function.npy'
# Output point cloud locations
PATH_OUP_XYZ = PATH + 'Heatsink_Output_XYZ.npy'
# Output point cloud values
PATH_OUP_U = PATH + 'Heatsink_Output_Function.npy'

# Dataset params
n_train = 1000
n_test = 100
n_total = n_train + n_test
# The number of points in (output) point cloud
n_points = 19517

# FNO configs
modes = 8
width = 20

# Training params
batch_size = 20
learning_rate = 0.001
epochs = 501
patience = epochs // 20     # scheduler

# Grid params
oversamp_ratio = 1.0        # used to calculate grid sizes
input_dim = 1               # (u0)
output_dim = 5              # (T, u, v, w, p)

# K-D tree params
n_subdomains = 16


################################################################
# Training and evaluation
################################################################
def main(train_a, train_u_sd, test_a, test_u_sd):
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_a, train_u_sd), 
        batch_size=batch_size, shuffle=True,
        generator=torch.Generator(device=device)
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_a, test_u_sd),
        batch_size=batch_size, shuffle=False,
        generator=torch.Generator(device=device)
    )

    model = FNO3d(modes, modes, modes, width, 
        in_channels=input_dim, 
        out_channels=output_dim*n_subdomains).cuda()
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
            out = model(x).reshape(batch_size, grid_shape[2], grid_shape[1], 
                grid_shape[0], output_dim, n_subdomains)\
                .permute(0, 5, 4, 1, 2, 3)\
                .reshape(-1, output_dim, 
                    grid_shape[2], grid_shape[1], grid_shape[0])
                # Output shape: (batch * n_subdomains, output_dim
                #   s3, s2, s1)

            # Interpolation (from grids to point cloud)
            u = F.grid_sample(input=out, grid=output_xyz_sd, 
                padding_mode='border', align_corners=False)
                # Output shape: (batch * n_subdomains, output_dim, 
                #   n_points_sd_padded, 1, 1)
            out = u.squeeze(-1).squeeze(-1).permute(0, 2, 1)\
                .reshape(batch_size, n_subdomains, -1, output_dim)\
                .permute(0, 2, 3, 1)
                # Output shape: (batch_size, n_points_sd_padded, 
                #   output_dim, n_subdomains)
            out = y_normalizer.decode(out)
            out = out * output_u_mask_sd
            out = out * output_u_sd_mask

            l2 = myloss(out.permute(0, 1, 3, 2), 
                y.permute(0, 1, 3, 2))
            l2.backward()

            optimizer.step()
            train_l2 += l2.item()

        scheduler.step(train_l2)

        model.eval()
        test_T_l2 = 0.0
        test_u_l2 = 0.0
        test_v_l2 = 0.0
        test_w_l2 = 0.0
        test_p_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x = model_encoder(x)
                out = model(x).reshape(batch_size, grid_shape[2], grid_shape[1], 
                    grid_shape[0], output_dim, n_subdomains)\
                    .permute(0, 5, 4, 1, 2, 3)\
                    .reshape(-1, output_dim, 
                        grid_shape[2], grid_shape[1], grid_shape[0])
                    # Output shape: (batch * n_subdomains, output_dim
                    #   s3, s2, s1)

                # Interpolation (from grids to point cloud)
                u = F.grid_sample(input=out, grid=output_xyz_sd, 
                    padding_mode='border', align_corners=False)
                    # Output shape: (batch * n_subdomains, output_dim, 
                    #   n_points_sd_padded, 1, 1)
                out = u.squeeze(-1).squeeze(-1).permute(0, 2, 1)\
                    .reshape(batch_size, n_subdomains, -1, output_dim)\
                    .permute(0, 2, 3, 1)
                    # Output shape: (batch_size, n_points_sd_padded, 
                    #   output_dim, n_subdomains)
                out = y_normalizer.decode(out)
                out = out * output_u_mask_sd
                out = out * output_u_sd_mask

                out, y = out.permute(0, 1, 3, 2), \
                    y.permute(0, 1, 3, 2)
                test_T_l2 += myloss(out[..., 0], 
                    y[..., 0], multi_channel=False).item()
                test_u_l2 += myloss(out[..., 1], 
                    y[..., 1], multi_channel=False).item()
                test_v_l2 += myloss(out[..., 2], 
                    y[..., 2], multi_channel=False).item()
                test_w_l2 += myloss(out[..., 3], 
                    y[..., 3], multi_channel=False).item()
                test_p_l2 += myloss(out[..., 4], 
                    y[..., 4], multi_channel=False).item()

        train_l2 /= n_train
        test_T_l2 /= n_test
        test_u_l2 /= n_test
        test_v_l2 /= n_test
        test_w_l2 /= n_test
        test_p_l2 /= n_test
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
    # Load data and preprocessing
    ################################################################
    input_xy = np.load(PATH_INP_XY)         # shape: (85, 2)
    input_a = np.load(PATH_INP_A)           # shape: (1100, 85, 1)
    output_xyz = np.load(PATH_OUP_XYZ)      # shape: (19517, 3)
    output_u = np.load(PATH_OUP_U)          # shape: (1100, 19517, 5)
    # Mask the point where u, v, w, p has no definition
    output_u_mask = np.where(output_u[0] == 0, 0, 1)
        # shape: (19517, 5)
    print("Start KD-Tree splitting...")
    t1 = default_timer()
    point_cloud = output_xyz.tolist()
    # Use kd-tree to generate subdomain division
    tree= KDTree(
        point_cloud, dim=3, n_subdomains=n_subdomains, 
        n_blocks=6**3, return_indices=True
    )
    tree.solve()
    # Gather subdomain info
    bbox_sd = tree.get_subdomain_bounding_boxes()
    indices_sd = tree.get_subdomain_indices()
    # Pad the point cloud of each subdomain to the same size
    max_n_points_sd = np.max([len(indices_sd[i]) 
        for i in range(n_subdomains)])
    output_xyz_sd = np.zeros((1, max_n_points_sd, n_subdomains, 3))
    output_u_sd = np.zeros((n_total, 
        max_n_points_sd, output_dim, n_subdomains))
    output_u_mask_sd = np.zeros((1, 
        max_n_points_sd, output_dim, n_subdomains))
    # Another mask is used to ignore padded zeros when calculating errors
    output_u_sd_mask = np.zeros((1, max_n_points_sd, 1, n_subdomains))
    # The grid shape
    grid_shape = [-1] * 3
        # (s1, s2, s3)
    for i in range(n_subdomains):
        # Normalize to [-1, 1]
        xy = output_xyz[indices_sd[i], :]
        _min, _max = np.min(xy, axis=0, keepdims=True), \
            np.max(xy, axis=0, keepdims=True)
        xy = (xy - _min) / (_max - _min) * 2 - 1
        # Long side alignment
        bbox = bbox_sd[i]
        scales = [bbox[j][1] - bbox[j][0] for j in range(3)]
        order = np.argsort(scales)
        xy = xy[:, order]
        # Calculate the grid shape
        _grid_shape = cal_grid_shape(
            oversamp_ratio * len(indices_sd[i]), scales)
        _grid_shape.sort()
        grid_shape = np.maximum(grid_shape, _grid_shape)
        # Applying
        output_xyz_sd[0, :len(indices_sd[i]), i, :] = xy
        output_u_sd[:, :len(indices_sd[i]), :, i] = \
            output_u[:, indices_sd[i], :]
        output_u_mask_sd[0, :len(indices_sd[i]), :, i] = \
            output_u_mask[indices_sd[i], :]
        output_u_sd_mask[0, :len(indices_sd[i]), 0, i] = 1.
    print(grid_shape)
    t2 = default_timer()
    print("Finish KD-Tree splitting, time elapsed: {:.1f}s".format(t2-t1))

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
        # shape: (ntotal, s3, s2, 1, 1)
    t2 = default_timer()
    print("Finish interpolation, time elapsed: {:.1f}s".format(t2-t1))

    output_xyz_sd = torch.from_numpy(output_xyz_sd).cuda().float()
    output_xyz_sd = output_xyz_sd.repeat([batch_size, 1, 1, 1])\
        .permute(0, 2, 1, 3)\
        .reshape(batch_size * n_subdomains, -1, 1, 1, 3)
        # shape: (batch * n_subdomains, n_points_sd_padded, 1, 1, 3)
    output_u_sd = torch.from_numpy(output_u_sd).float()
        # shape: (ntotal, n_points_sd_padded, output_dim, n_subdomains)
    output_u_mask_sd = torch.from_numpy(output_u_mask_sd).cuda().float()
        # shape: (1, n_points_sd_padded, output_dim, n_subdomains)
    output_u_sd_mask = torch.from_numpy(output_u_sd_mask).cuda().float()
        # shape: (1, n_points_sd_padded, 1, n_subdomains)
    input_a_grid = torch.from_numpy(input_a_grid).float()
        # shape: (ntotal, s3, s2, 1, 1)

    train_a = input_a_grid[:n_train].cuda()
    test_a = input_a_grid[-n_test:].cuda()

    train_u_sd = output_u_sd[:n_train].cuda()
    test_u_sd = output_u_sd[-n_test:].cuda()

    a_normalizer = UnitGaussianNormalizer(train_a)
    train_a = a_normalizer.encode(train_a)
    test_a = a_normalizer.encode(test_a)

    y_normalizer = UnitGaussianNormalizer(train_u_sd)

    ################################################################
    # Re-experiment with different random seeds
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
            main(train_a, train_u_sd, test_a, test_u_sd)
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
