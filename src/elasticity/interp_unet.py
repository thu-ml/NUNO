"""
Reference
----------
author:   Zongyi Li and Daniel Zhengyu Huang
source:   https://raw.githubusercontent.com/zongyi-li/Geo-FNO
reminder: slightly modified, e.g., file path, better output format, etc.
"""

import torch.nn.functional as F
from timeit import default_timer
from util.utilities import *
from torch.optim import Adam

################################################################
# UNet
################################################################
""" UNET model: https://github.com/milesial/Pytorch-UNet"""
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
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
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = nn.Linear(n_channels, 32)
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
        x1 = self.inc(x).permute(0,3,1,2)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = x.permute(0,2,3,1)
        x = self.outc(x)
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
PATH = 'data/elasticity/'
INPUT_PATH = PATH+'Random_UnitCell_mask_10_interp.npy'
PATH_Sigma = PATH+'Random_UnitCell_sigma_10.npy'
PATH_XY = PATH+'Random_UnitCell_XY_10.npy'
Ntotal = 2000
ntrain = 1000
ntest = 200

batch_size = 20
learning_rate = 0.001

epochs = 501
step_size = 100
gamma = 0.5

modes = 12
width = 32

r = 1
h = int(((41 - 1) / r) + 1)
s = h


################################################################
# training and evaluation
################################################################
def main(x_train, x_test, train_s, test_s, train_xy, test_xy):
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, train_xy, train_s), 
        batch_size=batch_size, shuffle=True,
        generator=torch.Generator(device=device)
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test, test_xy, test_s), 
        batch_size=batch_size, shuffle=False,
        generator=torch.Generator(device=device)
    )

    model = UNet().cuda()
    print(count_params(model))

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    myloss = LpLoss(size_average=False)
    t0 = default_timer()
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_l2 = 0
        for x, xy, sigma in train_loader:
            x = x.cuda()
            mask = x.clone()

            optimizer.zero_grad()
            out = model(x)
            out = out*mask

            # Interpolation (from grids to point cloud)
            # Normalize to [-1, 1]
            _min, _max = torch.min(xy, dim=1, keepdim=True)[0], torch.max(xy, dim=1, keepdim=True)[0]
            xy = (xy - _min) / (_max - _min) * 2 - 1
            xy = xy.unsqueeze(-2)
                # Output shape: (batch, n_points, 1, 2)
            u = F.grid_sample(input=out.permute(0, 3, 2, 1), grid=xy, 
                padding_mode='border', align_corners=False)
                # Output shape: (batch, 1, n_points, 1)
            u = u.squeeze(-1).permute(0, 2, 1)
                # Output shape: (batch, n_points, 1)

            loss = myloss(u.reshape(batch_size, -1), sigma.view(batch_size, -1))
            loss.backward()

            optimizer.step()
            train_l2 += loss.item()

        scheduler.step()

        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, xy, sigma in test_loader:
                x = x.cuda()
                mask = x.clone()

                out = model(x)
                out2 = out * mask

                # Interpolation (from grids to point cloud)
                # Normalize to [-1, 1]
                _min, _max = torch.min(xy, dim=1, keepdim=True)[0], torch.max(xy, dim=1, keepdim=True)[0]
                xy = (xy - _min) / (_max - _min) * 2 - 1
                xy = xy.unsqueeze(-2)
                    # Output shape: (batch, n_points, 1, 2)
                u = F.grid_sample(input=out2.permute(0, 3, 2, 1), grid=xy, 
                    padding_mode='border', align_corners=False)
                    # Output shape: (batch, 1, n_points, 1)
                u = u.squeeze(-1).permute(0, 2, 1)
                    # Output shape: (batch, n_points, 1)

                test_l2 += myloss(u.reshape(batch_size, -1), sigma.view(batch_size, -1)).item()

        train_l2 /= ntrain
        test_l2 /= ntest

        t2 = default_timer()
        print("[Epoch {}] Time: {:.1f}s L2: {:>4e} Test_L2: {:>4e}"
                .format(ep, t2-t1, train_l2, test_l2))

    # Return final results
    return train_l2, test_l2, t2-t0

if __name__ == "__main__":

    ################################################################
    # load data and data normalization
    ################################################################
    input = np.load(INPUT_PATH)
    input = torch.tensor(input, dtype=torch.float).permute(2,0,1)
    input_s = np.load(PATH_Sigma)
    input_s = torch.tensor(input_s, dtype=torch.float).permute(1,0).unsqueeze(-1)
    input_xy = np.load(PATH_XY)
    input_xy = torch.tensor(input_xy, dtype=torch.float).permute(2,0,1)

    x_train = input[:Ntotal][:ntrain, ::r, ::r][:, :s, :s]
    x_test = input[:Ntotal][-ntest:, ::r, ::r][:, :s, :s]

    x_train = x_train.reshape(ntrain, s, s, 1)
    x_test = x_test.reshape(ntest, s, s, 1)

    train_s = input_s[:ntrain]
    test_s = input_s[-ntest:]
    train_xy = input_xy[:ntrain]
    test_xy = input_xy[-ntest:]


    ################################################################
    # re-experiment with different random seeds
    ################################################################
    train_l2_res = []
    test_l2_res = []
    time_res = []
    for i in range(5):
        print("=== Round %d ==="%(i+1))
        set_random_seed(SEED_LIST[i])
        train_l2, test_l2, time = main(x_train, x_test, 
            train_s, test_s, train_xy, test_xy)
        train_l2_res.append(train_l2)
        test_l2_res.append(test_l2)
        time_res.append(time)
    print("=== Finish ===")
    for i in range(5):
        print("[Round {}] Time: {:.1f}s Train_L2: {:>4e} Test_L2: {:>4e}"
                .format(i+1, time_res[i], train_l2_res[i], test_l2_res[i]))

