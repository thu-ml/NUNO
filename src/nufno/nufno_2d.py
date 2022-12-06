import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchkbnufft as tkbn
from util.utilities3 import device

EPSILON = 1e-6

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

        # adjoint NUFFT and NUFFT
        self.adj_nufft_ob = tkbn.KbNufftAdjoint(
            im_size=(self.modes1 * 2 + 1, self.modes2 * 2 + 1),
            device=device
        )
        self.nufft_ob = tkbn.KbNufft(
            im_size=(self.modes1 * 2 + 1, self.modes2 * 2 + 1),
            device=device
        )

        scale = (1 / n_subdomains)
        self.weights1_subdomain = nn.Parameter(scale * 
            torch.rand(n_subdomains, self.modes1 * 2 + 1, 
                self.modes2, dtype=torch.cfloat))
        self.weights2_subdomain = nn.Parameter(scale * 
            torch.rand(n_subdomains, self.modes1 + 1, 
                1, dtype=torch.cfloat))
        scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(scale * 
            torch.rand(in_channels, out_channels, 
                self.modes1 * 2 + 1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * 
            torch.rand(in_channels, out_channels, 
                self.modes1 + 1, 1, dtype=torch.cfloat))
        scale = (1 / n_subdomains)
        self.weights1_subdomain_back = nn.Parameter(scale * 
            torch.rand(n_subdomains, self.modes1 * 2 + 1, 
                self.modes2, dtype=torch.cfloat))
        self.weights2_subdomain_back = nn.Parameter(scale * 
            torch.rand(n_subdomains, self.modes1 + 1, 
                1, dtype=torch.cfloat))

    # Aggregate subdomains
    def compl_mul2d_subdomain(self, input, weights):
        # (batch, in_channel, n_subdomains, x, y), (n_subdomains, x, y) -> (batch, in_channel, x, y)
        return torch.einsum("bisxy,sxy->bixy", input, weights)

    # Recover subdomains
    def compl_mul2d_subdomain_back(self, input, weights):
        # (batch, out_channel, x, y), (n_subdomains, x, y) -> (batch, out_channel, n_subdomains, x, y)
        return torch.einsum("bixy,sxy->bisxy", input, weights)

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x, loc, ind, sep):
        batchsize = x.shape[0]
        # Go to frequency space
        # remember the scale of data in each subdomain
        data_scale = {} # (b, i) -> (min_val, max_val)
        # Adjoint NUFFT (from point cloud to grid)
        imag_1_batch = []
        imag_2_batch = []
        for b in range(batchsize):
            imag_1_subdomain = []
            imag_2_subdomain = []
            for i in range(self.n_subdomains):
                indices = ind[b, sep[b, i]:sep[b, i+1]]
                data = x[b, :, indices]
                data_scale[(b, i)] = (
                    torch.min(data, dim=-1, keepdim=True)[0], 
                    torch.max(data, dim=-1, keepdim=True)[0]
                )
                data = data.unsqueeze(1) + 0j
                omega = loc[b, indices, :].permute(1, 0)
                # Normalize to [-pi, pi)
                omega = omega - omega.min(1, keepdim=True)[0]
                omega = omega / (omega.max(1, keepdim=True)[0] + EPSILON)
                omega = omega * 2 * np.pi - np.pi
                # adjoint NUFTT
                imag = self.adj_nufft_ob(data.conj(), omega, norm="ortho").conj()
                imag = imag.squeeze(1) # (in_channel, modes1 * 2 + 1, modes2 * 2 + 1)
                # Because we are handing real signal,
                # we can just store one half of the result (the other half is its complex conjugate)
                imag_1_subdomain.append(imag[:, :, :self.modes2]) 
                imag_2_subdomain.append(imag[:, :self.modes1+1, self.modes2:self.modes2+1])
            imag_1_batch.append(
                torch.stack(imag_1_subdomain, dim=1)
            )
            imag_2_batch.append(
                torch.stack(imag_2_subdomain, dim=1)
            )
        # Shape: (batch, in_channel, n_subdomains, modes1 * 2 + 1, modes2)
        imag_1 = torch.stack(imag_1_batch)
        # Shape: (batch, in_channel, n_subdomains, modes1 + 1, 1)
        imag_2 = torch.stack(imag_2_batch)

        # Aggregate subdomain Fourier results
        # Shape: (batch, in_channel, modes1 * 2 + 1, modes2)
        imag_1 = self.compl_mul2d_subdomain(imag_1, self.weights1_subdomain)
        # Shape: (batch, in_channel, modes1 + 1, 1)
        imag_2 = self.compl_mul2d_subdomain(imag_2, self.weights2_subdomain)

        # Perform linear transform
        # Shape: (batch, out_channel, modes1 * 2 + 1, modes2)
        imag_1 = self.compl_mul2d(imag_1, self.weights1)
        # Shape: (batch, out_channel, modes1 + 1, 1)
        imag_2 = self.compl_mul2d(imag_2, self.weights2)

        # Recover subdomain Fourier results
        # Shape: (batch, out_channel, n_subdomains, modes1 * 2 + 1, modes2)
        imag_1 = self.compl_mul2d_subdomain_back(imag_1, self.weights1_subdomain_back)
        # Shape: (batch, out_channel, n_subdomains, modes1 + 1, 1)
        imag_2 = self.compl_mul2d_subdomain_back(imag_2, self.weights2_subdomain_back)

        # First assemble the complete frequency matrix
        # Shape: (batch, out_channel, n_subdomains, modes1 * 2 + 1, modes2 * 2 + 1)
        imag = torch.zeros((batchsize, self.out_channels, self.n_subdomains, self.modes1 * 2 + 1, self.modes2 * 2 + 1), dtype=torch.cfloat)
        imag[:, :, :, :self.modes1 * 2 + 1, :self.modes2] = imag_1
        imag[:, :, :, -self.modes1 * 2 - 1:, -self.modes2:] = imag_1.flip([-1,-2]).conj()
        imag[:, :, :, -self.modes1 - 1:, self.modes2:self.modes2 + 1] = imag_2.flip([-1,-2]).conj()
        imag[:, :, :, :self.modes1 + 1, self.modes2:self.modes2 + 1] = imag_2
        
        # Return to physical space
        # NUFFT (from grid to point cloud)
        output = torch.zeros((batchsize, self.out_channels, x.shape[-1]))
        for b in range(batchsize):
            for i in range(self.n_subdomains):
                indices = ind[b, sep[b, i]:sep[b, i+1]]
                imag_ = imag[b:b+1, :, i, :, :].permute(1, 0, 2, 3)
                omega = loc[b, indices, :].permute(1, 0)
                # Normalize to [-pi, pi)
                omega = omega - omega.min(1, keepdim=True)[0]
                omega = omega / (omega.max(1, keepdim=True)[0] + EPSILON)
                omega = omega * 2 * np.pi - np.pi
                # NUFTT
                imag_ = self.nufft_ob(imag_.conj(), omega, norm="ortho").conj()
                imag_ = imag_.real.squeeze(1)
                # Rescale to [min_val, max_val]
                min_val, max_val = data_scale[(b, i)]
                imag_ = imag_ - imag_.min(1, keepdim=True)[0]
                imag_ = imag_ / (imag_.max(1, keepdim=True)[0] + EPSILON)
                imag_ = imag_ * (max_val - min_val) + min_val
                
                output[b, :, indices] = imag_

        # Return the result
        # Shape: (batch, out_channel, n_points)
        return output


class NUFNO2d(nn.Module):
    def __init__(self, Nx, Ny, modes1, modes2, 
        width, dim_in=1, dim_out=1, n_subdomains=1):
        """
        Non-Uniform Fourier Neural Operator. 
        1. The Fourier Transform is replaced by 
        Non-uniform discrete Fourier Transform.
        2. The dimension of input parameters can
        vary from output function values.
        3. Non-uniform query locations are allowed.
        
        input1: `x`, the parameters (a_1, a_2, ...)
        input1 shape: (batchsize, Nx, 1)

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
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.n_subdomains = n_subdomains
        if Nx != Ny:
            self.fc_embed = nn.Linear(Nx, Ny) # lift to match
        self.fc0 = nn.Linear(self.dim_in + 2, self.width) # input channel is 3: (a or a's embeddings, x, y)

        self.conv0 = NUSpectralConv2d(
            self.width, self.width, self.modes1, self.modes2,
            self.n_subdomains
        )
        self.conv1 = NUSpectralConv2d(
            self.width, self.width, self.modes1, self.modes2,
            self.n_subdomains
        )
        self.conv2 = NUSpectralConv2d(
            self.width, self.width, self.modes1, self.modes2,
            self.n_subdomains
        )
        self.conv3 = NUSpectralConv2d(
            self.width, self.width, self.modes1, self.modes2,
            self.n_subdomains
        )
        # Conv1d instead of Conv2d since our input is a point cloud
        # rather than an image
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

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
        if self.Nx != self.Ny:
            x = x.permute(0, 2, 1)
            x = self.fc_embed(x)
            x = x.permute(0, 2, 1)
        x = torch.cat((x, loc), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x1 = self.conv0(x, loc, ind, sep)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x, loc, ind, sep)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x, loc, ind, sep)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x, loc, ind, sep)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x
