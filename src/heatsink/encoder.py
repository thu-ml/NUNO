import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, width=32, target_size=16):
        super(Encoder, self).__init__()
        """
        Encode 2D images to 3D (volumetric) images
        """
        self.width = width
        self.target_size = target_size
        self.fc0 = nn.Linear(1, self.width)
        self.fc1 = nn.Linear(self.width, 4*self.width)
        self.fc2 = nn.Linear(4*self.width, 4*self.width)
        self.fc3 = nn.Linear(4*self.width, target_size)
        self.activation = torch.tanh

    def forward(self, x):
        # Input: x (batch, s3, s2, 1, channel)
        x = x.permute(0, 1, 2, 4, 3)
        x = self.fc0(x)
        x = self.activation(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        x = x.permute(0, 1, 2, 4, 3)

        # Input: x (batch, s3, s2, target_size, channel)
        return x
