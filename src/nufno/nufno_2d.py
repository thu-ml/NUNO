import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..elasticity import geofno


################################################################
# Spectral Convolution Layer
################################################################
class SpectralConv2d(geofno.SpectralConv2d):
    
    def forward_noft(self, x_ft):
        '''
        Perform spectral convolution for inputs which live in frequency
        domain instead of physical domain
        '''
        # Note: we do not run FFT since x_ft is already in frequency space
        x_ft1 = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        x_ft2 = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(x_ft.shape[0], self.out_channels, 
            self.s1, self.s2 // 2 + 1, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1, :self.modes2] = x_ft1
        out_ft[:, :, -self.modes1:, :self.modes2] = x_ft2

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(self.s1, self.s2))
        return x


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
    Non-Uniform 2D Fourier Neural Network 
    It contains two branches:
    1. the main branch takes point clouds as inputs,
    which are then passed through 5 Fourier layers.
    The first layer maps the point cloud into an latent image
    via a NUFFT (whose implementation is a naive DFT, the same as 
    in Geo-FNO), while the last layer maps the latent image back
    to the point cloud via a INUFFT.
    2. another branch takes the frequency modes of subdomain images as inputs
    (which is interpolated from the point cloud to the subdomain grid). 
    These images are passed through several Fourier layers independently. 
    The outputs are then combined to a global image and merged into main branch.
    This branch is expected to learn the local features.

    Input:
    1. `u_sd`, frequency modes of subdomain images.
    Shape: (batchsize, n_subdomains, modes1, modes2, dim_in).
    2. `sd_info`, subdomain information (xmin, ymin, xlen, ylen).
    Shape: (batchsize, n_subdomains, 4).
    3. `xy`, coordinates of the point cloud.
    Shape: (batchsize, n_points, 2).

    Output:
    Prediction of values of the target function at the point cloud.
    Shape: (batchsize, n_points, dim_out).
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

        # Auxiliary branch
        # Extract local features from subdomain images and
        # compensate the interpolation errors in the main branch
        sd_n_channels = self.n_channels // self.n_subdomains
        self.sd_layer0 = FourierLayer(self.dim_in + 2, sd_n_channels, self.modes1, self.modes2)
        self.sd_layer1 = FourierLayer(sd_n_channels, sd_n_channels, self.modes1, self.modes2)

        self.sd_conv0 = SpectralConv2d(self.dim_in, sd_n_channels, self.modes1, self.modes2, 
            self.width1, self.width2)
        self.sd_conv1 = SpectralConv2d(sd_n_channels, sd_n_channels, self.modes1, self.modes2,
            self.width1, self.width2)
        self.sd_conv2 = SpectralConv2d(sd_n_channels, sd_n_channels, self.modes1, self.modes2,
            self.width1, self.width2)

        # Main branch
        # Conventional Fourier layers, extract global features
        self.layer0 = FourierLayer(self.n_channels, self.n_channels, self.modes1, self.modes2, 
            self.width1, self.width2, transform=False, merging=True, merge_channels=sd_n_channels)
        self.layer1 = FourierLayer(self.n_channels, self.n_channels, self.modes1, self.modes2, 
            merging=True, merge_channels=sd_n_channels)
        self.layer2 = FourierLayer(self.n_channels, self.n_channels, self.modes1, self.modes2, 
            dropout=True, merging=True, merge_channels=sd_n_channels)
        self.layer3 = FourierLayer(self.n_channels, self.n_channels, self.modes1, self.modes2)
        self.layer4 = FourierLayer(self.n_channels, self.n_channels, self.modes1, self.modes2, 
            activation=False, transform=False, merging=True, merge_channels=2, merge_1d=True)

        # Fully-connected layers
        self.fc0 = nn.Linear(2, self.n_channels)
        self.fc1 = nn.Linear(self.n_channels, 128)
        self.fc2 = nn.Linear(128, self.dim_out)


    def forward(self, u_sd, sd_info, xy):
        # u_sd: (batch, n_subdomains, m1, m2, dim_in)
        # sd_info: (batch, n_subdomains, 4)
        # xy: (batch, n_points, 2)
        # modes1, modes2 are abbreviated as m1, m2
        # width1, width2 are abbreviated as w1, w2

        # Auxiliary branch
        u1, u2, u3 = self.auxiliary_branch(u_sd, sd_info)

        u = self.fc0(xy)
        u = u.permute(0, 2, 1)

        u = self.layer0(u, u1, x_in=xy)
        u = self.layer1(u, u2)
        u = self.layer2(u, u3)
        u = self.layer3(u)
        u = self.layer4(u, xy.permute(0, 2, 1), x_out=xy)

        u = u.permute(0, 2, 1)
        u = self.fc1(u)
        u = F.gelu(u)
        u = self.fc2(u)
        # u: (batch, n_points, dim_out)
        return u

    def auxiliary_branch(self, u_sd, sd_info):
        batchsize = u_sd.shape[0]
        u_sd = u_sd.permute(0, 1, 4, 2, 3)
            # (batch, n_subdomains, dim_in, w1, w2)

        u1 = self.combine_subdomain_result(u_sd, sd_info)
        u1 = self.sd_conv0.forward_noft(u1)

        u2 = torch.fft.irfft2(u_sd, s=(self.width1, self.width2))\
            .permute(0, 1, 3, 4, 2)
        grid = self.get_subdomain_grid(
            [batchsize, self.n_subdomains, self.width1, self.width2], sd_info)
            # (batch, n_subdomains, w1, w2, 2)
        u2 = torch.cat((u2, grid), dim=-1)\
                .permute(0, 1, 4, 2, 3)\
                .reshape(-1, self.dim_in + 2, self.width1, self.width2)
            # (batch * n_subdomains, dim_in + 2, w1, w2)
        tmp = self.sd_layer0(u2)
            # (batch * n_subdomains, sd_nchannels, w1, w2)
        u2 = tmp.reshape(batchsize, self.n_subdomains, 
                -1, self.width1, self.width2)
        u2 = self.combine_subdomain_result(u2, sd_info, make_ft=True)
        u2 = self.sd_conv1.forward_noft(u2)

        u3 = self.sd_layer1(tmp)
            # (batch * n_subdomains, sd_nchannels, w1, w2)
        u3 = u3.reshape(batchsize, self.n_subdomains, 
                -1, self.width1, self.width2)
        u3 = self.combine_subdomain_result(u3, sd_info, make_ft=True)
        u3 = self.sd_conv2.forward_noft(u3)

        return u1, u2, u3

    def combine_subdomain_result(self, u_sd, sd_info, make_ft=False):
        '''
        Combine the frequency modes of all the subdomains to
        obtain the global frequency mode.
        '''
        # u_sd: (batchsize, n_subdomains, channel, w1, w2) or
        #   (batchsize, n_subdomains, channel, m1, m2)
        if make_ft:
            u_sd = torch.fft.rfft2(u_sd)
            u_sd = torch.concat((
                u_sd[:, :, :, :self.modes1, :self.modes2],
                u_sd[:, :, :, -self.modes1:, :self.modes2]
            ), dim=-2)

        # Frequency number (m1, m2)
        m1 = u_sd.shape[-2]
        m2 = u_sd.shape[-1]
        ky =  torch.cat((torch.arange(start=0, end=self.modes1, step=1), \
                            torch.arange(start=-(self.modes1), end=0, step=1)), 0).reshape(m1,1).repeat(1,m2)
        kx =  torch.arange(start=0, end=self.modes2, step=1).reshape(1,m2).repeat(m1,1)
        
        # Combination coefficients
        # exp(-j * 2pi * (xmin/tot_xlen * kx + ymin/tot_ylen * ky)),
        # where tot_xlen == tot_ylen == 1 for this case
        kx = kx.reshape(1, 1, m1, m2)
        ky = ky.reshape(1, 1, m1, m2)

        xmin, ymin = sd_info[..., 0:1], sd_info[..., 1:2]
        xmin, ymin = xmin.unsqueeze(-1).repeat(1, 1, m1, m2), \
            ymin.unsqueeze(-1).repeat(1, 1, m1, m2)

        coef = torch.exp(-1j * 2 * np.pi * (xmin * kx + ymin * ky)).to(torch.cfloat)
        
        # (batch, n_subdomains, channel, m1, m2), (batch, n_subdomains, m1, m2) 
        # -> (batch, channel, m1, m2)
        u_ft = torch.einsum("bncxy,bnxy->bcxy", u_sd, coef)
        return u_ft

    def get_subdomain_grid(self, _shape, sd_info):
        '''
        Generate grid coordinates for each subdomain.
        '''
        # grid: (batch, n_subdomains, w1, w2, 2)
        batchsize, n_subdomains, w1, w2 = _shape
        xmin, ymin, xlen, ylen = \
            sd_info[:, :, 0:1], sd_info[:, :, 1:2], sd_info[:, :, 2:3], sd_info[:, :, 3:4]
        gridx = torch.linspace(0, 1, w1, dtype=torch.float)\
            .reshape(1, 1, w1).repeat([batchsize, n_subdomains, 1])
        gridx = gridx * xlen + xmin
        gridx = gridx.unsqueeze(-1).expand(-1, -1, -1, w2)
        gridy = torch.linspace(0, 1, w2, dtype=torch.float)\
            .reshape(1, 1, w2).repeat([batchsize, n_subdomains, 1])
        gridy = gridy * ylen + ymin
        gridy = gridy.unsqueeze(-2).expand(-1, -1, w1, -1)
        return torch.stack((gridx, gridy), dim=-1)
