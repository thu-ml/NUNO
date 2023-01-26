from timeit import default_timer
import torch.nn.functional as F
from src.kdtree.tree import KDTree
from util.utilities import *
from .interp_fno import FNO3d
from scipy.interpolate import LinearNDInterpolator, RBFInterpolator


################################################################
# Configs
################################################################
# Data path
PATH = 'data/channel/'
# Point cloud locations 
PATH_XY = PATH + 'Channel_Flow_XY.npy'
# Point cloud values (u, v, p)
PATH_U = PATH + 'Channel_Flow_Velocity_Pressure.npy'
# Point cloud values (u, v, p) in each subdomain
PATH_U_SD = PATH + \
    'Preprocess_Channel_Flow_Velocity_Pressure_Subdomain.npy'
# Mask for ignore padded zeros in each subdomain
PATH_U_SD_M = PATH + \
    'Preprocess_Channel_Flow_Velocity_Pressure_Subdomain_Mask.npy'
# Grid values (u, v, p) in each subdomain
PATH_U_SD_G = PATH + \
    'Preprocess_Channel_Flow_Velocity_Pressure_Subdomain_Grid.npy'

# Dataset params
n_train = 1000
n_test = 200
n_total = n_train + n_test
# The number of points in point cloud
n_points = 3809

# FNO configs
modes = 8
width = 20

# Training params
batch_size = 20
learning_rate = 0.001
epochs = 501
patience = epochs // 20     # scheduler

# Grid params
oversamp_ratio = 1.5        # used to calculate grid sizes
T_in = 15                   # input: [0, 0.15)
T = 30                      # output: [0.15, 0.30)
output_dim = 3              # (u, v, p)

# K-D tree params
n_subdomains = 8

# Wether to save or load preprocessing results
SAVE_PREP = False
LOAD_PREP = False


################################################################
# Training and evaluation
################################################################
def main(train_a_sd, train_u_sd, test_a_sd, test_u_sd):
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

    model = FNO3d(modes, modes, modes, width, 
        in_channels=output_dim*n_subdomains, 
        out_channels=output_dim*n_subdomains).cuda()
    print(count_params(model))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=patience)

    myloss = MultiLpLoss(size_average=False)
    y_normalizer.cuda()
    t0 = default_timer()
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_l2 = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            out = model(x).reshape(batch_size, s1_padded, s2_padded, 
                (T-T_in), output_dim, n_subdomains)\
                .permute(0, 5, 1, 2, 3, 4)\
                .reshape(-1, s1_padded, s2_padded, (T-T_in) * 3)
                # Output shape: (batch * n_subdomains, 
                #   s1_padded, s2_padded, (T-T_in) * 3)

            # Interpolation (from grids to point cloud)
            u = F.grid_sample(input=out.permute(0, 3, 1, 2), grid=input_xy_sd, 
                padding_mode='border', align_corners=False)
                # Output shape: (batch * n_subdomains, (T-T_in) * 3, 
                #   n_points_sd_padded, 1)
            out = u.squeeze(-1).permute(0, 2, 1)\
                .reshape(batch_size, n_subdomains, -1, T-T_in, 3)\
                .permute(0, 2, 3, 4, 1)
                # Output shape: (batch_size, n_points_sd_padded, 
                #   T-T_in, 3, n_subdomains)
            out = y_normalizer.decode(out)
            out = out * input_u_sd_mask

            l2 = myloss(out.permute(0, 1, 2, 4, 3), 
                y.permute(0, 1, 2, 4, 3))
            l2.backward()

            optimizer.step()
            train_l2 += l2.item()

        scheduler.step(train_l2)

        model.eval()
        test_u_l2 = 0.0
        test_v_l2 = 0.0
        test_p_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                out = model(x).reshape(batch_size, s1_padded, s2_padded, 
                    (T-T_in), output_dim, n_subdomains)\
                    .permute(0, 5, 1, 2, 3, 4)\
                    .reshape(-1, s1_padded, s2_padded, (T-T_in) * 3)
                    # Output shape: (batch * n_subdomains, 
                    #   s1_padded, s2_padded, (T-T_in) * 3)

                # Interpolation (from grids to point cloud)
                u = F.grid_sample(input=out.permute(0, 3, 1, 2), grid=input_xy_sd, 
                    padding_mode='border', align_corners=False)
                    # Output shape: (batch * n_subdomains, 
                    #   (T-T_in) * 3, n_points_sd_padded, 1)
                out = u.squeeze(-1).permute(0, 2, 1)\
                    .reshape(batch_size, n_subdomains, -1, T-T_in, 3)\
                    .permute(0, 2, 3, 4, 1)
                    # Output shape: (batch_size, n_points_sd_padded, 
                    #   T-T_in, 3, n_subdomains)
                out = y_normalizer.decode(out)
                out = out * input_u_sd_mask

                test_u_l2 += myloss(out[..., 0, :], 
                    y[..., 0, :], multi_channel=False).item()
                test_v_l2 += myloss(out[..., 1, :], 
                    y[..., 1, :], multi_channel=False).item()
                test_p_l2 += myloss(out[..., 2, :], 
                    y[..., 2, :], multi_channel=False).item()

        train_l2 /= n_train
        test_u_l2 /= n_test
        test_v_l2 /= n_test
        test_p_l2 /= n_test
        test_l2 = (test_u_l2+test_v_l2+test_p_l2)/3

        t2 = default_timer()
        print("[Epoch {}] Time: {:.1f}s L2: {:>4e} Test_L2: {:>4e}"
                    .format(ep, t2-t1, train_l2, test_l2))

    # Return final results
    return train_l2, test_l2, t2-t0, test_u_l2, test_v_l2, test_p_l2


if __name__ == "__main__":
    ################################################################
    # Load data and preprocessing
    ################################################################
    input_xy = np.load(PATH_XY)            # shape: (3809, 2)
    input_u = np.load(PATH_U)              # shape: (1200, 3809, 31, 3)

    print("Start KD-Tree splitting...")
    t1 = default_timer()
    point_cloud = input_xy.tolist()
    # Use kd-tree to generate subdomain division
    tree= KDTree(
        point_cloud, dim=2, n_subdomains=n_subdomains, 
        n_blocks=8, return_indices=True
    )
    tree.solve()
    # Gather subdomain info
    bbox_sd = tree.get_subdomain_bounding_boxes()
    indices_sd = tree.get_subdomain_indices()
    input_xy_sd = np.zeros((np.max([len(indices_sd[i]) 
        for i in range(n_subdomains)]), n_subdomains, 2))
    for i in range(n_subdomains):
        # Normalize to [-1, 1]
        xy = input_xy[indices_sd[i], :]
        _min, _max = np.min(xy, axis=0, keepdims=True), \
            np.max(xy, axis=0, keepdims=True)
        xy = (xy - _min) / (_max - _min) * 2 - 1
        # Long side alignment
        bbox = bbox_sd[i]
        if bbox[0][1] - bbox[0][0] < bbox[1][1] - bbox[1][0]:
            xy = np.flip(xy, axis=1)
        input_xy_sd[:len(indices_sd[i]), i, :] = xy
    t2 = default_timer()
    print("Finish KD-Tree splitting, time elapsed: {:.1f}s".format(t2-t1))

    if LOAD_PREP:
        input_u_sd_grid = np.load(PATH_U_SD_G)   
            # shape: (1200, s1_padded, s2_padded, 31, 3, n_subdomains) 
        input_u_sd = np.load(PATH_U_SD)          
            # shape: (1200, n_points_sd_padded, 31, 3, n_subdomains) 
        input_u_sd_mask = np.load(PATH_U_SD_M)   
            # shape: (1, n_points_sd_padded, 1, 1, n_subdomains) 
    else:
        t1 = default_timer()
        print("Start interpolation...")
        # Calculate the padded grid size
        max_grid_size_x, max_grid_size_y = -1, -1
        grid_sizes = []
        is_transposed = [False] * n_subdomains
        for i in range(n_subdomains):
            n_points = len(indices_sd[i])
            bbox = bbox_sd[i]
            # Calculate the grid size, where the aspect ratio of the discrete grid 
            # remains the same as the that of the original subdomain (bbox)
            grid_size_x = np.sqrt(n_points * oversamp_ratio * \
                (bbox[0][1] - bbox[0][0]) / (bbox[1][1] - bbox[1][0]))
            grid_size_y = grid_size_x * (bbox[1][1] - bbox[1][0]) / (bbox[0][1] - bbox[0][0])
            grid_size_x, grid_size_y = max(int(np.round(grid_size_x)), 2), \
                max(int(np.round(grid_size_y)), 2)
            grid_sizes.append((grid_size_x, grid_size_y))
            # Long side alignment to reduce paddings
            if bbox[0][1] - bbox[0][0] < bbox[1][1] - bbox[1][0]:
                grid_size_x, grid_size_y = grid_size_y, grid_size_x
                is_transposed[i] = True
            max_grid_size_x, max_grid_size_y = max(max_grid_size_x, 
                grid_size_x), max(max_grid_size_y, grid_size_y)

        # Interpolation from point cloud to uniform grid
        input_u_sd_grid = []
        point_cloud = input_xy
        point_cloud_val = np.transpose(input_u, (1, 2, 3, 0)) 
        interp_linear = LinearNDInterpolator(point_cloud, point_cloud_val)
        interp_rbf = RBFInterpolator(point_cloud, point_cloud_val, neighbors=6)
        for i in range(n_subdomains):
            grid_size_x, grid_size_y = grid_sizes[i]
            bbox = bbox_sd[i]
            # Linear interpolation
            grid_x = np.linspace(bbox[0][0], bbox[0][1], num=grid_size_x)
            grid_y = np.linspace(bbox[1][0], bbox[1][1], num=grid_size_y)
            grid_x, grid_y = np.meshgrid(grid_x, grid_y)
            grid_val = interp_linear(grid_x, grid_y)
            # Fill nan values
            nan_indices = np.isnan(grid_val)[..., 0, 0, 0]
            fill_vals = interp_rbf(np.stack((grid_x[nan_indices], grid_y[nan_indices]), axis=1))
            grid_val[nan_indices] = fill_vals
            # Resize to the same size via FFT-IFFT
            freq = np.fft.rfft2(grid_val, axes=(0, 1))
            s1_padded, s2_padded = max_grid_size_y, max_grid_size_x 
            if is_transposed[i]:
                s1_padded, s2_padded = s2_padded, s1_padded
            square_freq = np.zeros((s1_padded, 
                s2_padded // 2 + 1, T+1, 3, n_total)) + 0j
            square_freq[:min(s1_padded//2, freq.shape[0]//2), 
                    :min(s2_padded//2+1, freq.shape[1]//2+1), ...] = \
                freq[:min(s1_padded//2, freq.shape[0]//2), 
                    :min(s2_padded//2+1, freq.shape[1]//2+1), ...]
            square_freq[-min(s1_padded//2, freq.shape[0]//2):, 
                    :min(s2_padded//2+1, freq.shape[1]//2+1), ...] = \
                freq[-min(s1_padded//2, freq.shape[0]//2):,
                    :min(s2_padded//2+1, freq.shape[1]//2+1), ...]
            grid_val = np.fft.irfft2(square_freq, 
                s=(s1_padded, s2_padded), axes=(0, 1))
            if is_transposed[i]:
                grid_val = np.transpose(grid_val, (1, 0, 2, 3, 4))
            input_u_sd_grid.append(np.transpose(grid_val, (4, 0, 1, 2, 3)))
        input_u_sd_grid = np.transpose(np.array(input_u_sd_grid), (1, 2, 3, 4, 5, 0))

        # Pad the point-cloud values of each subdomain to the same size
        # Mask is used to ignore padded zeros when calculating errors
        input_u_sd = np.zeros((n_total, 
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
        .reshape(batch_size * n_subdomains, -1, 1, 2)
        # shape: (batch * n_subdomains, n_points_sd_padded, 1, 2)

    s1_padded, s2_padded = input_u_sd_grid.shape[1:3]
    input_u_sd_grid = torch.from_numpy(
        input_u_sd_grid.reshape(n_total, s1_padded, s2_padded, T+1, -1)).float()
    input_u_sd = torch.from_numpy(input_u_sd).float()
    input_u_sd_mask = torch.from_numpy(input_u_sd_mask).cuda().float()

    train_a_sd = input_u_sd_grid[:n_train, ..., :T_in, :].cuda()
    test_a_sd = input_u_sd_grid[-n_test:, ..., :T_in, :].cuda()

    train_u_sd = input_u_sd[:n_train, ..., T_in:T, :, :].cuda()
    test_u_sd = input_u_sd[-n_test:, ..., T_in:T, :, :].cuda()

    a_normalizer = UnitGaussianNormalizer(train_a_sd)
    train_a_sd = a_normalizer.encode(train_a_sd)
    test_a_sd = a_normalizer.encode(test_a_sd)

    y_normalizer = UnitGaussianNormalizer(train_u_sd)

    ################################################################
    # Re-experiment with different random seeds
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
            main(train_a_sd, train_u_sd, test_a_sd, test_u_sd)
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
