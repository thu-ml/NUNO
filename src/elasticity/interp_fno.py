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
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
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

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(3, self.width)  # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0, self.padding, 0, self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
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

    model = FNO2d(modes, modes, width).cuda()
    print(count_params(model))

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    myloss = LpLoss(size_average=False)

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
    return train_l2, test_l2

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
    for i in range(5):
        print("=== Round %d ==="%(i+1))
        set_random_seed(SEED_LIST[i])
        train_l2, test_l2 = main(x_train, x_test, 
            train_s, test_s, train_xy, test_xy)
        train_l2_res.append(train_l2)
        test_l2_res.append(test_l2)
    print("=== Finish ===")
    for i in range(5):
        print("[Round {}] Train_L2: {:>4e} Test_L2: {:>4e}"
                .format(i+1, train_l2_res[i], test_l2_res[i]))
