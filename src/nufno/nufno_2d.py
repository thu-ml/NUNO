import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.modes1 = modes1    # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x
    
    def forward_ft(self, x_ft, width1, width2):
        x_ft1 = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        x_ft2 = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(x_ft.shape, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1, :self.modes2] = x_ft1
        out_ft[:, :, -self.modes1:, :self.modes2] = x_ft2

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(width1, width2))
        return x


class NUFNO2d(nn.Module):
    def __init__(self, modes1=16, modes2=16, width1=32, width2=32, 
        n_channels_global=32, n_channels_local=8, n_subdomains=32, dim_in=1, dim_out=1):
        super(NUFNO2d, self).__init__()

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
        self.width1 = width1
        self.width2 = width2
        self.n_channels_global = n_channels_global
        self.n_channels_local = n_channels_local
        self.n_subdomains = n_subdomains
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.padding = 9 # pad the domain if input is non-periodic

        # Lift the modes (complex number)
        scale = 2 / np.sqrt(dim_in)
        self.fc0_weights = nn.Parameter(
            scale * (torch.rand(self.dim_in, self.n_channels_global, dtype=torch.cfloat) - (0.5 + 0.5j)))
        self.fc0_bias = nn.Parameter(
            scale * (torch.rand(1, 1, 1, self.n_channels_global, dtype=torch.cfloat) - (0.5 + 0.5j)))
        self.fc4_weights = nn.Parameter(
            scale * (torch.rand(self.dim_in, self.n_channels_local, dtype=torch.cfloat) - (0.5 + 0.5j)))
        self.fc4_bias = nn.Parameter(
            scale * (torch.rand(1, 1, 1, self.n_channels_local, dtype=torch.cfloat) - (0.5 + 0.5j)))

        self.conv_global0 = SpectralConv2d(self.n_channels_global, self.n_channels_global, self.modes1, self.modes2)
        self.conv_global1 = SpectralConv2d(self.n_channels_global, self.n_channels_global, self.modes1, self.modes2)
        self.conv_global2 = SpectralConv2d(self.n_channels_global, self.n_channels_global, self.modes1, self.modes2)
        self.conv_local0 = SpectralConv2d(self.n_channels_local, self.n_channels_local, self.modes1, self.modes2)
        self.conv_local1 = SpectralConv2d(self.n_channels_local, self.n_channels_local, self.modes1, self.modes2)
        self.conv_local2 = SpectralConv2d(self.n_channels_local, self.n_channels_local, self.modes1, self.modes2)

        self.w_global1 = nn.Conv2d(self.n_channels_global, self.n_channels_global, 1)
        self.w_global2 = nn.Conv2d(self.n_channels_global, self.n_channels_global, 1)
        self.w_local1 = nn.Conv2d(self.n_channels_local, self.n_channels_local, 1)
        self.w_local2 = nn.Conv2d(self.n_channels_local, self.n_channels_local, 1)

        self.fc1 = nn.Linear(self.n_channels_global, self.n_channels_local)
        self.fc2 = nn.Linear(self.n_channels_local, 128)
        self.fc3 = nn.Linear(128, self.dim_out)

    def complex_lift(self, x_ft, weight, bias):
        # (batch, x, y, in_channel), (in_channel, out_channel) -> (batch, x, y, out_channel)
        return torch.einsum("bxyi,io->bxyo", x_ft, weight) + bias
    
    
    def combine_subdomain_modes(self, x_ft_sd, sd_info):
        m1 = x_ft_sd.shape[-3]
        m2 = x_ft_sd.shape[-2]

        # Frequency number (m1, m2)
        ky =  torch.cat((torch.arange(start=0, end=self.modes1, step=1), \
                            torch.arange(start=-(self.modes1), end=0, step=1)), 0).reshape(m1,1).repeat(1,m2)
        kx =  torch.arange(start=0, end=self.modes2, step=1).reshape(1,m2).repeat(m1,1)
        
        # Combination coefficients
        # exp(-j * 2pi * (xlen/tot_xlen * kx + ylen/tot_ylen * ky)),
        # where tot_xlen == tot_ylen == for this case
        kx = kx.reshape(1, 1, m1, m2)
        ky = ky.reshape(1, 1, m1, m2)
        xlen = sd_info[:, :, 2:3].unsqueeze(-1).repeat(1, 1, m1, m2)
        ylen = sd_info[:, :, 3:4].unsqueeze(-1).repeat(1, 1, m1, m2)
        coef = torch.exp(-1j * 2 * np.pi * (xlen * kx + ylen * ky)).to(torch.cfloat)
        
        # (batch, n_subdomains, m1, m2, dim_in), (batch, n_subdomains, m1, m2) -> (batch, m1, m2, dim_in)
        return torch.einsum("bnxyc,bnxy->bxyc", x_ft_sd, coef)


    def forward(self, u_sd, xy, ind, sep, sd_info):
        # u_sd: (batch, n_subdomains, m1, m2, dim_in)
        u = self.combine_subdomain_modes(u_sd, sd_info)
        u = self.complex_lift(u, self.fc0_weights, self.fc0_bias)    
                                    # (batch, m1, m2, n_channels_global)
        u = u.permute(0, 3, 1, 2)   # (batch, n_channels_global, m1, m2)

        # Global
        u = self.conv_global0.forward_ft(u, self.width1, self.width2)
        u = F.gelu(u)

        u1 = self.conv_global1(u)
        u2 = self.w_global1(u)
        u = u1 + u2
        u = F.gelu(u)

        u1 = self.conv_global2(u)
        u2 = self.w_global2(u)
        u = u1 + u2
        u = F.gelu(u)               # (batch, n_channels_global, w1, w2)

        # Local
        u = u.permute(0, 2, 3, 1)   # (batch, w1, w2, n_channels_global)
        u = self.fc1(u)             # (batch, w1, w2, n_channels_local)
        
        u = u.permute(0, 3, 1, 2)   # (batch, n_channels_local, m1, m2)
        u_ft = torch.fft.rfft2(u)
        u_ft = torch.concat((
            u_ft[:, :, :self.modes1, :self.modes2],
            u_ft[:, :, -self.modes1:, :self.modes2]), dim=-2)
                                    # (batch, n_channels_local, m1, m2)
        u_ft = torch.repeat_interleave(u_ft, self.n_subdomains, dim=0)
                                    # (batch * n_subdomains, n_channels_local, m1, m2)
        u = self.complex_lift(u_sd.reshape(-1, u_sd.shape[2], u_sd.shape[3], self.dim_in),
            self.fc4_weights, self.fc4_bias).permute(0, 3, 1, 2) + u_ft
        u = self.conv_local0.forward_ft(u, self.width1, self.width2)
        u = F.gelu(u)

        u1 = self.conv_local1(u)
        u2 = self.w_local1(u)
        u = u1 + u2
        u = F.gelu(u)

        u1 = self.conv_local2(u)
        u2 = self.w_local2(u)
        u = u1 + u2 # (batch * n_subdomains, n_channels_local, w1, w2)

        # Interpolation (from subdomain grids to point cloud)
        sd_xy = self.get_subdomains_xy(xy, ind, sep, sd_info)
            # Output shape: (batch * n_subdomains, n_points, 1, 2)
        u = F.grid_sample(input=u, grid=sd_xy, padding_mode='border', align_corners=False)
            # Output shape: (batch * n_subdomains, n_channels_local, n_points, 1)
        u = u.squeeze(-1).permute(0, 2, 1)
            # Output shape: (batch * n_subdomains, n_points, n_channels_local)
        u = self.gather_subdomains(u, ind, sep)
        
        u = self.fc2(u)
        u = F.gelu(u)
        u = self.fc3(u)

        return u
    
    def get_subdomains_xy(self, xy, ind, sep, sd_info):
        # ind (batch, Nx) the input value
        # sep (batch, N_subdomains+1) the input value
        # subdomain_info (batch, N_subdomains, 7) the input value
        # Statics
        batch_size = xy.shape[0]
        max_n_points = torch.max(sd_info[:, :, -1]).to(torch.int)
        # Padding
        sd_xy = torch.zeros(batch_size * self.n_subdomains, max_n_points, 2)
        for b in range(batch_size):
            for j in range(self.n_subdomains):
                indices = ind[b, sep[b, j]:sep[b, j+1]]
                loc = xy[b, indices, :]
                # Normalize to [-1, 1]
                _min, _len = sd_info[b, j:j+1, :2], sd_info[b, j:j+1, 2:4]
                loc = (loc - _min) / _len * 2 - 1
                sd_xy[b * self.n_subdomains + j, :len(indices), :] = loc
        
        return sd_xy.unsqueeze(-2)
    
    def gather_subdomains(self, u_sd, ind, sep):
        # u (batch, Nx, width) the input value
        # ind (batch, Nx) the input value
        # sep (batch, N_subdomains+1) the input value
        # Statics
        batch_size = ind.shape[0]
        u = torch.zeros(batch_size, ind.shape[1], self.n_channels_local)
        for b in range(batch_size):
            for j in range(self.n_subdomains):
                indices = ind[b, sep[b, j]:sep[b, j+1]]
                u[b, indices, :] = \
                    u_sd[b * self.n_subdomains + j, :len(indices), :]

        return u
