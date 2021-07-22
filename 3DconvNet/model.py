import torch
import torch.nn as nn
from torch.nn.modules.conv import Conv3d

class ConvModel(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ConvModel, self).__init__()
        # conv3d (N, C, D, H, W)
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=64, kernel_size=7, padding=1)
        self.bn1 = nn.BatchNorm3d(num_features=64)
        self.relu = nn.ReLU()
        # square window stride=2
        self.mpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(num_features=128)
        self.conv3 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(num_features=256)
        self.conv4 = nn.Conv3d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(num_features=512)
        self.conv5 = nn.Conv3d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm3d(1024)
        self.fc = nn.Linear(4333568, num_classes) # input perceptrons calculated based on test example

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.mpool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

def test():
    model = ConvModel(in_channels=3, num_classes=2)
    x = torch.randn(1,3,20,50,50)
    with torch.no_grad():
        y = model(x)
    print(y.size())

test()