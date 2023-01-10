import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .geofno import SpectralConv2d


################################################################
# Fourier Layer
################################################################
class FourierLayer(nn.Module):
    '''
    The Fourier layer contains a spectral convolution layer, 
    and a spatial linear transform.
    Note: compared with original implementation of FNO, we
    add a Dropout layer.
    '''
    def __init__(self, in_channels, out_channels, modes1, modes2, 
        width1=32, width2=32, dropout=False, dropout_rate=0.1, activation=True, 
        transform=True, merging=False, merge_channels=None, merge_1d=False):
        '''
        `transform`: bool, optional
        Whether to perform linear transform in the physical space.
        The default is `True`.
        `merging`: bool, optional
        Whether to merge external inputs from another branch.
        The default is `False`.
        `merge_channels`: int, optional
        The number of channels of the external inputs.
        The default is `None` which means it equals to `in_channels`.
        Note: this value is valid if `merging==True`.
        `merge_1d`: bool, optional
        Whether the external inputs are 1d sequences or 2d images.
        The default is `False` which means they are 2d images.
        '''
        super(FourierLayer, self).__init__()
        self.dropout = dropout
        self.activation = activation
        self.transform = transform
        self.merging = merging
        self.conv = SpectralConv2d(in_channels, out_channels, 
            modes1, modes2, width1, width2)
        if transform:
            self.w = nn.Conv2d(in_channels, out_channels, 1)
        if merging:
            if merge_1d:
                self.b = nn.Conv1d(
                    merge_channels if merge_channels is not None 
                    else in_channels, out_channels, 1)
            else:
                self.b = nn.Conv2d(
                    merge_channels if merge_channels is not None 
                    else in_channels, out_channels, 1)
        if dropout:
            self.d = nn.Dropout(dropout_rate)

    def forward(self, x, external_input=None, x_in=None, x_out=None):
        '''
        `x_in`, `x_out`: the point coordinates (when input or output live
        on a point cloud).
        Note: please make sure `merging==True` if `external_input` is
        specified.
        '''
        y = self.conv(x, x_in=x_in, x_out=x_out)
        if self.transform:
            y = y + self.w(x)
        if external_input is not None:
            y = y + self.b(external_input)
        if self.activation:
            y = F.gelu(y)
        if self.dropout:
            y = self.d(y)
        return y

class NUFNO2d(nn.Module):
    """
    Non-Uniform 2D Fourier Neural Network\n
    Compared with the vanilla FNO, we add a Fourier layer
    for interpolating the subdomain images back to point cloud. 

    Input:
    1. `u`, the density distribution of the point cloud.
    Shape: (batch_size, width1, width2, dim_in).
    2. `xy_sd`, coordinates of point clouds in each subdomain.
    Shape: (batch_size, n_subdomains, max_sd_n_points, 2).

    Output:
    Prediction at point clouds in each subdomain.
    Shape: (batch_size, n_subdomains, max_sd_n_points, dim_out).
    """
    def __init__(self, modes1=16, modes2=16, width1=32, width2=32, 
        n_channels=32, n_subdomains=32, dim_in=1, dim_out=1):
        super(NUFNO2d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width1 = width1
        self.width2 = width2
        self.n_channels = n_channels
        self.n_subdomains = n_subdomains
        self.dim_in = dim_in
        self.dim_out = dim_out

        # Fourier layers
        self.layer0 = FourierLayer(self.n_channels, self.n_channels, self.modes1, self.modes2)
        self.layer1 = FourierLayer(self.n_channels, self.n_channels, self.modes1, self.modes2)
        self.layer2 = FourierLayer(self.n_channels, self.n_channels, self.modes1, self.modes2, 
            dropout=True)
        self.layer3 = FourierLayer(self.n_channels, self.n_channels, self.modes1, self.modes2)
        # For interpolation
        self.layer4 = FourierLayer(self.n_channels, self.n_channels, self.modes1, self.modes2,
            dropout=True)
        self.layer5 = FourierLayer(self.n_channels, self.n_channels, self.modes1, self.modes2, 
            activation=False, transform=False, merging=True, merge_channels=2, merge_1d=True)

        # Fully-connected layers
        self.fc0 = nn.Linear(self.dim_in + 2, self.n_channels)
        self.fc1 = nn.Linear(self.n_channels, 128)
        self.fc2 = nn.Linear(128, self.dim_out)
        # For interpolation
        self.fc_interp0 = nn.Linear(self.n_channels, 128)
        self.fc_interp1 = nn.Linear(128, self.n_channels * self.n_subdomains)


    def forward(self, u, xy_sd):
        # u: (batch, w1, w2, dim_in)
        # xy_sd: (batch, n_subdomains, max_sd_n_points, 2)
        # Note:
        # width1, width2 are abbreviated as w1, w2
        batch_size = u.shape[0]

        grid = self.get_grid(u.shape, u.device)
            # (batch, w1, w2, 2)
        u = torch.cat((u, grid), dim=-1)
        u = self.fc0(u)
        u = u.permute(0, 3, 1, 2)

        u = self.layer0(u)
        u = self.layer1(u)
        u = self.layer2(u)
        u = self.layer3(u)

        # Interpolation back to point cloud
        xy_sd = xy_sd.reshape(batch_size * self.n_subdomains, -1, 2)
        u = u.permute(0, 2, 3, 1)
        u = self.fc_interp0(u)
        u = F.gelu(u)
        u = self.fc_interp1(u).\
            reshape(-1, self.width1, self.width2, self.n_channels, self.n_subdomains).\
            permute(0, 4, 3, 1, 2).\
            reshape(-1, self.n_channels, self.width1, self.width2)
        u = self.layer4(u)
        u = self.layer5(u, xy_sd.permute(0, 2, 1), x_out=xy_sd)

        u = u.permute(0, 2, 1)
            # (batch_size * n_subdomains, max_sd_n_points, n_channels)
        u = self.fc1(u)
        u = F.gelu(u)
        u = self.fc2(u)
        # u: (batch_size * n_subdomains, max_sd_n_points, dim_out)

        return u.reshape(batch_size, self.n_subdomains, -1, self.dim_out)

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
