import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torchkbnufft as tkbn
from util.utilities3 import device

EPSILON = 1e-6

################################################################
# (Uniform) Fourier Layer 
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


################################################################
# Nou-Uniform Fourier Layer
################################################################

class NUSpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, 
        modes1, modes2, n_subdomains):
        """
        Non-Uniform 2D Fourier layer. 
        It does adjoint NUFFT (from point cloud to grid), 
        linear transform, and NUFFT (from grid back to point cloud).    
        """
        super(NUSpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 # Number of Fourier modes to preserve
        self.modes2 = modes2
        self.n_subdomains = n_subdomains

        # NUFFT
        self.nufft_ob = tkbn.KbNufft(
            im_size=(self.modes1 * 2 + 1, self.modes2 * 2 + 1),
            device=device
        )

        scale = (1 / (self.n_subdomains * out_channels))
        self.weights = nn.Parameter(scale * 
            torch.rand(self.n_subdomains, out_channels))
        self.bias = nn.Parameter(scale * 
            torch.rand(self.n_subdomains, out_channels))
        scale = (1 / (in_channels * self.n_subdomains * out_channels))
        self.weights1 = nn.Parameter(scale * 
            torch.rand(in_channels, self.n_subdomains * out_channels, 
                self.modes1 * 2 + 1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * 
            torch.rand(in_channels, self.n_subdomains * out_channels, 
                self.modes1 + 1, 1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x, loc, ind, sep):
        batchsize = x.shape[0]
        # Go to frequency space (odd len for simplicity)
        overall_ft = torch.fft.fftshift(torch.fft.fft2(
            x, norm="ortho", s=[
                x.size(-2) + 1 - (x.size(-2) % 2),
                x.size(-1) + 1 - (x.size(-1) % 2)
            ]
        )) # (batch, in_channel, kx, ky)
        ft_origin = (overall_ft.shape[-2] // 2, overall_ft.shape[-1] // 2)
        overall_ft_1 = overall_ft[:, :, 
            (ft_origin[0]-self.modes1):(ft_origin[0]+self.modes1+1), 
            (ft_origin[1]-self.modes2):ft_origin[1]]
        overall_ft_2 = overall_ft[:, :,
            (ft_origin[0]-self.modes1):(ft_origin[0]+1), 
            ft_origin[1]:(ft_origin[1]+1)]
        
        # Multiply relevant Fourier modes
        # Shape: (batch, n_subdomains * out_channel, kx, ky)
        imag_1 = self.compl_mul2d(overall_ft_1, self.weights1)
        imag_2 = self.compl_mul2d(overall_ft_2, self.weights2)
        
        # Return to physical space
        # First assemble the complete frequency matrix
        # Shape: (batch, n_subdomains * out_channel, modes1 * 2 + 1, modes2 * 2 + 1)
        images = torch.zeros((batchsize, self.out_channels * self.n_subdomains, self.modes1 * 2 + 1, self.modes2 * 2 + 1), dtype=torch.cfloat)
        images[:, :, :self.modes1 * 2 + 1, :self.modes2] = imag_1
        images[:, :, -self.modes1 * 2 - 1:, -self.modes2:] = imag_1.flip([-1,-2]).conj()
        images[:, :, -self.modes1 - 1:, self.modes2:self.modes2 + 1] = imag_2.flip([-1,-2]).conj()
        images[:, :, :self.modes1 + 1, self.modes2:self.modes2 + 1] = imag_2
        images[:, :, self.modes1, self.modes2] = images[:, :, self.modes1, self.modes2].real + 0j # Hermitian request
        images = images.reshape(batchsize * self.n_subdomains * self.out_channels, self.modes1 * 2 + 1, self.modes2 * 2 + 1)
        images = images.unsqueeze(1)

        # Then prepare point locations
        omegas = []
        for b in range(batchsize):
            for i in range(self.n_subdomains):
                indices = ind[b, sep[b, i]:sep[b, i+1]]
                # Prepare omega (normalized locations)
                omega = loc[b, indices, :] # (length x 2)
                # Normalize to [-pi, pi)
                omega = omega - omega.min(0, keepdim=True)[0]
                omega = omega / (omega.max(0, keepdim=True)[0] + EPSILON)
                omega = omega * 2 * np.pi - np.pi
                for _ in range(self.out_channels):
                    omegas.append(omega)
        # Padding to form batches
        omegas = pad_sequence(omegas, batch_first=True) # (num x length x 2)
        omegas = omegas.permute(0, 2, 1)
        
        # NUFFT (from grid to point cloud)
        # from timeit import default_timer
        # t1 = default_timer()
        values = self.nufft_ob(images.conj(), omegas, norm="ortho").conj()
        values = values.real.squeeze(1) # (num x length)
        # print("HIT", default_timer() - t1)
        # Place the values in the output
        output = torch.zeros((batchsize, self.out_channels, loc.shape[1]))
        for b in range(batchsize):
            for i in range(self.n_subdomains):
                indices = ind[b, sep[b, i]:sep[b, i+1]]
                res = values[
                    b * self.n_subdomains * self.out_channels +
                        i * self.out_channels:
                    b * self.n_subdomains * self.out_channels +
                        (i+1) * self.out_channels, :len(indices)]
                res = res * self.weights[i] + self.bias[i]
                output[b, :, indices] = res

        # Return the result
        # Shape: (batch, out_channel, n_points)
        return output


class NUFNO2d(nn.Module):
    def __init__(self, Nx, Ny, modes1, modes2, width,
        hidden_dim_x, hidden_dim_y, dim_in=1, dim_out=1, n_subdomains=1):
        """
        Non-Uniform Fourier Neural Operator. 
        1. The Fourier Transform is replaced by 
        Non-uniform discrete Fourier Transform.
        2. The dimension of input parameters can
        vary from output function values.
        3. Non-uniform query locations are allowed.
        
        input1: `x`, the parameters (a_1, a_2, ...)
        input1 shape: (batchsize, Nx, dim_in)

        input2: `loc`, query locations ((x_1, y_1), ...)
        input2 shape: (batchsize, Ny, 2)

        input3: `ind` and `sep`, the indices of the points in each subdomain.
        E.g., `ind=[1, 3, 5, 0, 2, 4]`, `sep=[0, 3, 6]` means
        the 2nd, 4th, 6th points in `loc` (ind[0:3]) belong to
        a subdomain while the others (ind[3:6]) belong to another subdomain.
        input3 shape: (batchsize, Ny) and (batchsize, n_subdomains + 1)

        output: the solution function at query locations
        output shape: (batchsize, Ny, dim_out)
        Note: the solution function is a scalar-valued function
        if `dim_out==1`, a vector-valued function if `dim_out>=2`.
        """
        super(NUFNO2d, self).__init__()
        self.Nx = Nx
        self.Ny = Ny
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.hidden_dim_x = hidden_dim_x # latent embedding (hidden_dim_x x hidden_dim_y)
        self.hidden_dim_y = hidden_dim_y # latent embedding (hidden_dim_x x hidden_dim_y)
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.n_subdomains = n_subdomains

        # From point cloud to grid embedding
        self.fc_embed_1 = nn.Linear(Nx, self.hidden_dim_x * self.hidden_dim_y)
        self.fc_embed_2 = nn.Linear(Ny, self.hidden_dim_x * self.hidden_dim_y)
        
        self.fc0 = nn.Linear(self.dim_in + 2, self.width) # input channel is 3: (a or a's embeddings, x, y)

        self.conv0 = SpectralConv2d(
            self.width, self.width, self.modes1, self.modes2
        )
        self.conv1 = SpectralConv2d(
            self.width, self.width, self.modes1, self.modes2
        )
        self.conv2 = SpectralConv2d(
            self.width, self.width, self.modes1, self.modes2
        )
        self.conv3 = SpectralConv2d(
            self.width, self.width, self.modes1, self.modes2
        )
        self.nuconv = NUSpectralConv2d(
            self.width, self.dim_out, self.modes1, self.modes2,
            self.n_subdomains
        )
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        # From grid embedding back to point cloud
        self.fc_deembed_1 = nn.Linear(2, self.width)
        self.fc_deembed_2 = nn.Linear(2, self.width)
        self.fc_deembed_3 = nn.Linear(self.dim_out, self.dim_out)
        self.fc_deembed_3 = nn.Linear(self.hidden_dim_x * self.hidden_dim_y, Ny)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, dim_out)

    def forward(self, x, loc, ind, sep):
        '''
        input1: `x`, the parameters (a_1, a_2, ...)
        input1 shape: (batchsize, Nx, dim_in)

        input2: `loc`, query locations ((x_1, y_1), ...)
        input2 shape: (batchsize, Ny, 2)

        input3: `ind` and `sep`, the indices of the points in each subdomain.
        E.g., `ind=[1, 3, 5, 0, 2, 4]`, `sep=[0, 3]` means
        the 2nd, 4th, 6th points in `loc` (ind[0:3]) belong to
        a subdomain while the others (ind[3:]) belong to another subdomain.
        input3 shape: (batchsize, Ny) and (batchsize, n_subdomains + 1)

        output: the solution function at query locations
        output shape: (batchsize, Ny, dim_out)
        Note: the solution function is a scalar-valued function
        if `dim_out==1`, a vector-valued function if `dim_out>=2`.
        '''
        # From point cloud to grid embedding
        batchsize = x.shape[0]
        x = x.permute(0, 2, 1)
        x = self.fc_embed_1(x)
        x = x.permute(0, 2, 1)
        loc_embed = loc.permute(0, 2, 1)
        loc_embed = self.fc_embed_2(loc_embed)
        loc_embed = loc_embed.permute(0, 2, 1)

        # Shape: (batchsize, hidden_dim_x * hidden_dim_y, dim_in + 2)
        x = torch.cat((x, loc_embed), dim=-1)
        x = x.reshape(batchsize, self.hidden_dim_x, self.hidden_dim_y, -1)
        x = self.fc0(x)
        # Shape: (batchsize, width, hidden_dim_x, hidden_dim_y)
        x = x.permute(0, 3, 1, 2)

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

        # From grid embedding back to point cloud (frequency space)
        # x1 = self.nuconv(x, loc, ind, sep)

        # From grid embedding back to point cloud (physical space)
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x) # (batchsize, hidden_dim_x, hidden_dim_y, dim_out)
        x = x.reshape(batchsize, self.hidden_dim_x*self.hidden_dim_y, 1)
        

        # grid = self.get_grid(x.shape, x.device)
        # Q = self.fc_deembed_1(loc) # batchsize x Ny x width
        # K = self.fc_deembed_2(grid.reshape(batchsize, self.hidden_dim_x * self.hidden_dim_y, 2)) # batchsize x (hidden_dim_x * hidden_dim_y) x width
        # V = self.fc_deembed_3(x.reshape(batchsize, self.hidden_dim_x * self.hidden_dim_y, self.dim_out)) # batchsize x (hidden_dim_x * hidden_dim_y) x dim_out
        # x2 = torch.matmul(Q, K.permute(0, 2, 1)) / np.sqrt(self.width) # batchsize x Ny x (hidden_dim_x * hidden_dim_y)
        # x2 = torch.softmax(x2, dim=-1)
        # x2 = torch.matmul(x2, V) # batchsize x Ny x dim_out
        # x = x2
        
        return self.fc_deembed_3(x.permute(0, 2, 1)).permute(0, 2, 1)
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
