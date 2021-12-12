"""
Standard AE encoder
"""

import torch
import torch.nn as nn


class Residual_block(nn.Module):
    """
    Create new Residual block
    Params:
        in_channels: Input channels
        hidden_inter: hidden channels for intermediate convolution
        hidden_final: Number of channels for output convolution
    """

    def __init__(self, in_channels, hidden_inter, hidden_final):
        super(Residual_block, self).__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_inter,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_inter,
                out_channels=hidden_final,
                kernel_size=1,
                stride=1,
                bias=False
            )
        )

    def forward(self, x):
        # Skip connection
        return x + self.net(x)


class Encoder(nn.Module):
    """
    Encoder block
    params:
        in_channels = input channels
        num_hidden = hidden blocks for encoder convolution
        residual_inter = intermediary residual block channels
    """

    def __init__(self, in_channels, num_hidden, residual_inter):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_hidden // 2,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=num_hidden // 2,
            out_channels=num_hidden,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.residual1 = Residual_block(
            in_channels=num_hidden,
            hidden_inter=residual_inter,
            hidden_final=num_hidden
        )
        self.residual2 = Residual_block(
            in_channels=num_hidden,
            hidden_inter=residual_inter,
            hidden_final=num_hidden
        )
        self.relu = nn.ReLU()
        self.apool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.residual1(x)
        x = self.residual2(x)
        x = self.apool(x)
        return x
