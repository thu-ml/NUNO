from timeit import default_timer
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from util.utilities import *
from scipy.interpolate import LinearNDInterpolator, \
    NearestNDInterpolator, RegularGridInterpolator

################################################################
# UNet 3D
################################################################
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet3d(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, bilinear=False):
        super(UNet3d, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = nn.Linear(n_channels+3, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = nn.Linear(32, n_classes)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x1 = self.inc(x).permute(0,4,1,2,3)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = x.permute(0,2,3,4,1)
        x = self.outc(x)
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
PATH_XYZ = 'data/heatsink/Heatsink_XYZ.npy'
PATH_U = 'data/heatsink/Heatsink_Function.npy'

ntrain = 900
ntest = 100
ntotal = ntrain + ntest
n_points = 19517

batch_size = 20
learning_rate = 0.001
epochs = 501
patience = epochs // 20
reg_lambda = 6e-3

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

    model = UNet3d(n_channels=input_dim, n_classes=output_dim).cuda()
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
            out = model(x)[..., :-2, :]
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
                out = model(x)[..., :-2, :]
                    # Output shape: (batch, s2, s1, s3, output_dim)
                out = y_normalizer.decode(out)

                test_l2 += myloss(out, y).item()

        train_l2 /= ntrain
        test_l2 /= ntest

        t2 = default_timer()
        print("[Epoch {}] Time: {:.1f}s L2: {:>4e} Test_L2: {:>4e}"
                    .format(ep, t2-t1, train_l2, test_l2))
    
    with torch.no_grad():
        pred = model(test_a_grid)[..., :-2, :]
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

    # s3 is too small for convolution
    # Increase by 2
    train_a_grid = torch.concat((train_a_grid, 
        train_a_grid[..., 0:2, :]), dim=-2)
    test_a_grid = torch.concat((test_a_grid, 
        test_a_grid[..., 0:2, :]), dim=-2)

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
