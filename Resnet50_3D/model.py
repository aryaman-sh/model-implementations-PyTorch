'''
RESNET50 for 3d images
'''

import torch
import torch.nn as nn

class ConvSection(nn.Module):
    """
    A convolution section
    """
    def __init__(self, in_channels, mid_channels, resize_conv=None, stride=1):
        super(ConvSection, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_channels)

        self.conv2 = nn.Conv3d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(mid_channels)

        self.conv3 = nn.Conv3d(mid_channels, mid_channels*4, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn3 = nn.BatchNorm3d(mid_channels*4)
        
        self.relu = nn.ReLU()
        self.resize_conv = resize_conv
        self.stride = stride

    def forward(self, x):
        idn = x.clone()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        
        if self.resize_conv is not None:
            idn = self.resize_conv(idn)
        
        x += idn
        x = self.relu(x)
        return x

class Resnet503D(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Resnet503D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU()
        self.mpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.apool = nn.AdaptiveAvgPool3d((1,1,1)) #TODO: set avgadaptive pool output
        
        self.layer_channel = 64
        self.conv2_x = self.convLayer(3, 64, 1)
        self.conv3_x = self.convLayer(4, 128, 2)
        self.conv4_x = self.convLayer(6, 256, 2)
        self.conv5_x = self.convLayer(3, 512, 2)
        
        self.fc = nn.Linear(512*4, num_classes)
        
    def convLayer(self, num_blocks, mid_channels, stride):
        layers = []
        resize_conv = None

        if stride != 1 or self.layer_channel != mid_channels*4:
            resize_conv = nn.Sequential(
                nn.Conv3d(self.layer_channel, mid_channels*4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(mid_channels*4),
            )
        layers.append(ConvSection(self.layer_channel, mid_channels, resize_conv, stride))
        self.layer_channel = mid_channels*4
        for i in range(num_blocks - 1):
            layers.append(ConvSection(self.layer_channel, mid_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.mpool(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.apool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

def test():
    model = Resnet503D(in_channels=1, num_classes=2)
    x = torch.randn(10,1,20,50,50)
    with torch.no_grad():
        y = model(x)
    print(y.size())

test()