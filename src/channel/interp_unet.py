from timeit import default_timer
from tqdm import tqdm
import torch.nn.functional as F
from util.utilities import *
from scipy.interpolate import LinearNDInterpolator, RBFInterpolator


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
PATH_XY = 'data/channel/Channel_Flow_XY.npy'
PATH_U = 'data/channel/Channel_Flow_Velocity_Pressure.npy'
PATH_U_G = 'data/channel/Preprocess_Channel_Flow_Velocity_Pressure_Grid.npy'

ntrain = 1000
ntest = 200
ntotal = ntrain + ntest
n_points = 3809

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


    model = UNet3d().cuda()
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
            out = model(x)[..., :-1, :].reshape(batch_size, S, S, -1)
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
                out = model(x)[..., :-1, :].reshape(batch_size, S, S, -1)
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

    # T_in is too small for convolution
    # Increase by 1
    train_a = torch.concat((train_a, train_a[..., 0:1, :]), dim=-2)
    test_a = torch.concat((test_a, test_a[..., 0:1, :]), dim=-2)

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

