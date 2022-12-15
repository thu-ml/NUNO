import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, s1=32, s2=32):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.s1 = s1
        self.s2 = s2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, u, x_in=None, x_out=None, iphi=None, code=None):
        batchsize = u.shape[0]

        # Compute Fourier coeffcients up to factor of e^(- something constant)
        if x_in == None:
            u_ft = torch.fft.rfft2(u)
            s1 = u.size(-2)
            s2 = u.size(-1)
        else:
            u_ft = self.fft2d(u, x_in, iphi, code)
            s1 = self.s1
            s2 = self.s2

        # Multiply relevant Fourier modes
        # print(u.shape, u_ft.shape)
        factor1 = self.compl_mul2d(u_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        factor2 = self.compl_mul2d(u_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        if x_out == None:
            out_ft = torch.zeros(batchsize, self.out_channels, s1, s2 // 2 + 1, dtype=torch.cfloat, device=u.device)
            out_ft[:, :, :self.modes1, :self.modes2] = factor1
            out_ft[:, :, -self.modes1:, :self.modes2] = factor2
            u = torch.fft.irfft2(out_ft, s=(s1, s2))
        else:
            out_ft = torch.cat([factor1, factor2], dim=-2)
            u = self.ifft2d(out_ft, x_out, iphi, code)

        return u
    
    def forward_ft(self, x_ft, width1, width2):
        x_ft1 = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        x_ft2 = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(x_ft.shape[0], self.out_channels, 
            width1, width2 // 2 + 1, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1, :self.modes2] = x_ft1
        out_ft[:, :, -self.modes1:, :self.modes2] = x_ft2

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(width1, width2))
        return x

    def fft2d(self, u, x_in, iphi=None, code=None):
        # u (batch, channels, n)
        # x_in (batch, n, 2) locations in [0,1]*[0,1]
        # iphi: function: x_in -> x_c

        batchsize = x_in.shape[0]
        N = x_in.shape[1]
        device = x_in.device
        m1 = 2 * self.modes1
        m2 = 2 * self.modes2 - 1

        # wavenumber (m1, m2)
        k_x1 =  torch.cat((torch.arange(start=0, end=self.modes1, step=1), \
                            torch.arange(start=-(self.modes1), end=0, step=1)), 0).reshape(m1,1).repeat(1,m2).to(device)
        k_x2 =  torch.cat((torch.arange(start=0, end=self.modes2, step=1), \
                            torch.arange(start=-(self.modes2-1), end=0, step=1)), 0).reshape(1,m2).repeat(m1,1).to(device)

        # print(x_in.shape)
   
        x = x_in


        # print(x.shape)
        # K = <y, k_x>,  (batch, N, m1, m2)
        K1 = torch.outer(x[...,0].view(-1), k_x1.view(-1)).reshape(batchsize, N, m1, m2)
        K2 = torch.outer(x[...,1].view(-1), k_x2.view(-1)).reshape(batchsize, N, m1, m2)
        K = K1 + K2

        # basis (batch, N, m1, m2)
        basis = torch.exp(-1j * 2 * np.pi * K).to(device)

        # Y (batch, channels, N)
        u = u + 0j
        Y = torch.einsum("bcn,bnxy->bcxy", u, basis)
        return Y

    def ifft2d(self, u_ft, x_out, iphi=None, code=None):
        # u_ft (batch, channels, kmax, kmax)
        # x_out (batch, N, 2) locations in [0,1]*[0,1]
        # iphi: function: x_out -> x_c

        batchsize = x_out.shape[0]
        N = x_out.shape[1]
        device = x_out.device
        m1 = 2 * self.modes1
        m2 = 2 * self.modes2 - 1

        # wavenumber (m1, m2)
        k_x1 =  torch.cat((torch.arange(start=0, end=self.modes1, step=1), \
                            torch.arange(start=-(self.modes1), end=0, step=1)), 0).reshape(m1,1).repeat(1,m2).to(device)
        k_x2 =  torch.cat((torch.arange(start=0, end=self.modes2, step=1), \
                            torch.arange(start=-(self.modes2-1), end=0, step=1)), 0).reshape(1,m2).repeat(m1,1).to(device)

        
        x = x_out

        # K = <y, k_x>,  (batch, N, m1, m2)
        K1 = torch.outer(x[:,:,0].view(-1), k_x1.view(-1)).reshape(batchsize, N, m1, m2)
        K2 = torch.outer(x[:,:,1].view(-1), k_x2.view(-1)).reshape(batchsize, N, m1, m2)
        K = K1 + K2

        # basis (batch, N, m1, m2)
        basis = torch.exp(1j * 2 * np.pi * K).to(device)

        # coeff (batch, channels, m1, m2)
        u_ft2 = u_ft[..., 1:].flip(-1, -2).conj()
        u_ft = torch.cat([u_ft, u_ft2], dim=-1)

        # Y (batch, channels, N)
        Y = torch.einsum("bcxy,bnxy->bcn", u_ft, basis)
        Y = Y.real
        return Y



class NUFNO2d(nn.Module):
    def __init__(self, modes1=16, modes2=16, width1=32, width2=32, 
        n_channels=32, n_subdomains=32, dim_in=1, dim_out=1):
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
        self.n_channels = n_channels
        self.n_subdomains = n_subdomains
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.padding = 9 # pad the domain if input is non-periodic

        # Encoder
        sd_n_channels = self.n_channels // self.n_subdomains
        self.encode_fc = nn.Linear(self.dim_in + 2, sd_n_channels) 
        self.encode_conv = SpectralConv2d(sd_n_channels, sd_n_channels, self.modes1, self.modes2)
        self.encode_w0 = nn.Conv2d(sd_n_channels, sd_n_channels, 1)
        self.amp_fc0 = nn.Linear(2, 32)     # (xlen, ylen) -> amplitude
        self.amp_fc1 = nn.Linear(32, 1)
        self.pha_fc0 = nn.Linear(2, 32)     # (xmin, ymin) -> phase
        self.pha_fc1 = nn.Linear(32, 2)
        self.decode_conv = SpectralConv2d(sd_n_channels, self.n_channels, self.modes1, self.modes2)

        self.conv0 = SpectralConv2d(self.n_channels, self.n_channels, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.n_channels, self.n_channels, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.n_channels, self.n_channels, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.n_channels, self.n_channels, self.modes1, self.modes2)
        self.conv4 = SpectralConv2d(self.n_channels, self.n_channels, self.modes1, self.modes2, self.width1, self.width2)

        self.w0 = nn.Conv2d(self.n_channels, self.n_channels, 1)
        self.w1 = nn.Conv2d(self.n_channels, self.n_channels, 1)
        self.w2 = nn.Conv2d(self.n_channels, self.n_channels, 1)
        self.w3 = nn.Conv2d(self.n_channels, self.n_channels, 1)

        self.b4 = nn.Conv1d(2, self.n_channels, 1)

        self.fc0 = nn.Linear(self.dim_in + 2, self.n_channels) 
            # each subdomain has 3 inputs: (a(x, y), x, y)
        self.fc1 = nn.Linear(self.n_channels, 128)
        self.fc2 = nn.Linear(128, self.dim_out)


    def forward(self, u_g, u_sd, sd_info, xy):
        # u_g: (batch, w1, w2, dim_in)
        # u_sd: (batch, n_subdomains, w1, w2, dim_in)
        # sd_info: (batch, n_subdomains, 4)
        # xy: (batch, n_points, 2)
        batchsize = u_g.shape[0]

        # Encode u_sd
        grid = self.get_subdomain_grid(u_sd.shape[:-1], sd_info)
        u_sd = torch.cat((u_sd, grid), dim=-1) 
            # (batch, n_subdomains, w1, w2, dim_in + 2)
        u_sd = self.encode_fc(u_sd).reshape(batchsize * self.n_subdomains, 
            self.width1, self.width2, -1)
            # (batch * n_subdomains, w1, w2, sd_n_channels)
        u_sd = u_sd.permute(0, 3, 1, 2)

        u_sd1 = self.encode_conv(u_sd)
        u_sd2 = self.encode_w0(u_sd)
        u_sd = u_sd1 + u_sd2
        u_sd = F.gelu(u_sd)

        u_sd = u_sd.reshape(batchsize, self.n_subdomains, 
            -1, self.width1, self.width2)
        u_sd = self.combine_subdomain_result(u_sd, sd_info)
            # (batch, n_channels, w1, w2)

        grid_g = self.get_global_grid(u_g.shape[:-1])
        u_g = torch.cat((u_g, grid_g), dim=-1)
        u_g = self.fc0(u_g)     # (batch, w1, w2, n_channels)
        u_g = u_g.permute(0, 3, 1, 2)
        # u = F.pad(u, [0, self.padding, 0, self.padding])

        u_g1 = self.conv0(u_g)
        u_g2 = self.w0(u_g)
        u_g = u_g1 + u_g2
        u_g = F.gelu(u_g)

        # Combine
        u = u_g + u_sd

        u1 = self.conv1(u)
        u2 = self.w1(u)
        u = u1 + u2
        u = F.gelu(u)

        u1 = self.conv2(u)
        u2 = self.w2(u)
        u = u1 + u2
        u = F.gelu(u)

        u1 = self.conv3(u)
        u2 = self.w3(u)
        u = u1 + u2
        u = F.gelu(u)

        u = self.conv4(u, x_out=xy)
        u3 = self.b4(xy.permute(0, 2, 1))
        u = u + u3

        # u = u[..., :-self.padding, :-self.padding]
        # u = u.permute(0, 2, 3, 1)
        u = u.permute(0, 2, 1)
        u = self.fc1(u)
        u = F.gelu(u)
        u = self.fc2(u)
        return u

    def combine_subdomain_result(self, u_sd, sd_info):
        # u_sd: batchsize, n_subdomains, sd_n_channel, w1, w2, 

        u_ft_sd = torch.fft.rfft2(u_sd)
        u_ft_sd = torch.concat((
            u_ft_sd[:, :, :, :self.modes1, :self.modes2],
            u_ft_sd[:, :, :, -self.modes1:, :self.modes2]
        ), dim=-2)

        m1 = u_ft_sd.shape[-2]
        m2 = u_ft_sd.shape[-1]

        # Frequency number (m1, m2)
        ky =  torch.cat((torch.arange(start=0, end=self.modes1, step=1), \
                            torch.arange(start=-(self.modes1), end=0, step=1)), 0).reshape(m1,1).repeat(1,m2)
        kx =  torch.arange(start=0, end=self.modes2, step=1).reshape(1,m2).repeat(m1,1)
        
        # Combination coefficients
        # exp(-j * 2pi * (xmin/tot_xlen * kx + ymin/tot_ylen * ky)),
        # where tot_xlen == tot_ylen == for this case
        # .unsqueeze(-1).repeat(1, 1, m1, m2)
        kx = kx.reshape(1, 1, m1, m2)
        ky = ky.reshape(1, 1, m1, m2)
        xymin = sd_info[:, :, 0:2]
        xylen = sd_info[:, :, 2:4]

        xymin = self.pha_fc0(xymin)
        xymin = F.tanh(xymin)
        xymin = self.pha_fc1(xymin)
        xmin, ymin = xymin[..., 0:1], xymin[..., 1:2]
        xmin, ymin = xmin.unsqueeze(-1).repeat(1, 1, m1, m2), \
            ymin.unsqueeze(-1).repeat(1, 1, m1, m2)

        amp = self.amp_fc0(xylen)
        amp = F.tanh(amp)
        amp = self.amp_fc1(amp)
        amp = amp.unsqueeze(-1).repeat(1, 1, m1, m2)

        coef = amp * torch.exp(-1j * (xmin * kx + ymin * ky)).to(torch.cfloat)
        
        # (batch, n_subdomains, c, m1, m2), (batch, n_subdomains, m1, m2) -> (batch, c, m1, m2)
        u_ft = torch.einsum("bncxy,bnxy->bcxy", u_ft_sd, coef)
        u = self.decode_conv.forward_ft(u_ft, self.width1, self.width2)

        return u


    def get_global_grid(self, _shape):
        batchsize, w1, w2 = _shape[0], _shape[1], _shape[2]
        gridx = torch.tensor(np.linspace(0, 1, w1), dtype=torch.float)
        gridx = gridx.reshape(1, w1, 1, 1).repeat([batchsize, 1, w2, 1])
        gridy = torch.tensor(np.linspace(0, 1, w2), dtype=torch.float)
        gridy = gridy.reshape(1, 1, w2, 1).repeat([batchsize, w1, 1, 1])
        return torch.cat((gridx, gridy), dim=-1)

    def get_subdomain_grid(self, _shape, sd_info):
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
