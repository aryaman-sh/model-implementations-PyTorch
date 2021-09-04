# Generator and Discriminator model
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, error_dim, image_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(error_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, image_dim),
            nn.Tanh(), # images are normalized in range -1,1
        )

    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, image_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(image_dim, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.net(x)
