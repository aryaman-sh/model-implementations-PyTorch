__author__ = "Aryaman Sharma"


class Conv1(torch.nn.Module):
    def __init__(self, in_channels=1, out_channels=64):
        super(Conv1, self).__init__()
        self.mpool = torch.nn.MaxPool3d(kernel_size=3, stride=2, padding=3)
        self.conv1 = torch.nn.Conv3d(in_channels=1, out_channels=64, stride=2, padding=3, kernel_size=7, bias=False)
        self.bn1 = torch.nn.BatchNorm3d(64)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x

class Conv2_x(torch.nn.Module):
    def __init__(self, in_channels=64, out_channels=64):
        super(Conv2_x, self).__init__()
        self.mpool = torch.nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.conv2_1 = torch.nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm3d(out_channels)
        self.relu = torch.nn.ReLU()
        self.conv2_2 = torch.nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm3d(out_channels)
        self.conv2_3 = torch.nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = torch.nn.BatchNorm3d(out_channels)
        self.conv2_4 = torch.nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = torch.nn.BatchNorm3d(out_channels)
        self.identity_downsample = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
            torch.nn.BatchNorm3d(out_channels)
        )

    def forward(self, x):
        idx = x.clone()
        idx = self.identity_downsample(idx)

        x = self.mpool(x)
        x = self.conv2_1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2_2(x)
        x = self.bn2(x)
        x += idx
        x = self.relu(x)
        idx2 = x.clone()
        x = self.conv2_3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv2_4(x)
        x = self.bn4(x)
        x += idx2
        x = self.relu(x)
        return x

class Conv3_x(torch.nn.Module):
    def __init__(self, in_channels=64, out_channels=128):
        super(Conv3_x, self).__init__()
        self.conv3_1 = torch.nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size= 3, stride=2, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm3d(out_channels)
        self.conv3_2 = torch.nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size= 3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm3d(out_channels)
        self.conv3_3 = torch.nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size= 3, stride=1, padding=1, bias=False)
        self.bn3 = torch.nn.BatchNorm3d(out_channels)
        self.conv3_4 = torch.nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size= 3, stride=1, padding=1, bias=False)
        self.bn4 = torch.nn.BatchNorm3d(out_channels)
        self.relu = torch.nn.ReLU()

        self.identity_downsample = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
            torch.nn.BatchNorm3d(out_channels)
        )

    def forward(self, x):
       
        idx = x.clone()
        idx = self.identity_downsample(idx)
        x = self.conv3_1(x)
       
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv3_2(x)
        x = self.bn2(x)
        x += idx
        x = self.relu(x)
        idx2 = x.clone()
        x = self.conv3_3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv3_4(x)
        x = self.bn4(x)
        x += idx2
        x = self.relu(x)
        return x

class Conv4_x(torch.nn.Module):
    def __init__(self, in_channels=128, out_channels=256):
        super(Conv4_x, self).__init__()
        self.conv3_1 = torch.nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size= 3, stride=2, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm3d(out_channels)
        self.conv3_2 = torch.nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size= 3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm3d(out_channels)
        self.conv3_3 = torch.nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size= 3, stride=1, padding=1, bias=False)
        self.bn3 = torch.nn.BatchNorm3d(out_channels)
        self.conv3_4 = torch.nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size= 3, stride=1, padding=1, bias=False)
        self.bn4 = torch.nn.BatchNorm3d(out_channels)
        self.relu = torch.nn.ReLU()

        self.identity_downsample = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
            torch.nn.BatchNorm3d(out_channels)
        )

    def forward(self, x):
        idx = x.clone()
        idx = self.identity_downsample(idx)
        x = self.conv3_1(x)
        
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv3_2(x)
        x = self.bn2(x)
        x += idx
        x = self.relu(x)
        idx2 = x.clone()
        x = self.conv3_3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv3_4(x)
        x = self.bn4(x)
        x += idx2
        x = self.relu(x)
        return x

class Conv5_x(torch.nn.Module):
    def __init__(self, in_channels=256, out_channels=512):
        super(Conv5_x, self).__init__()
        self.conv3_1 = torch.nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size= 3, stride=2, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm3d(out_channels)
        self.conv3_2 = torch.nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size= 3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm3d(out_channels)
        self.conv3_3 = torch.nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size= 3, stride=1, padding=1, bias=False)
        self.bn3 = torch.nn.BatchNorm3d(out_channels)
        self.conv3_4 = torch.nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size= 3, stride=1, padding=1, bias=False)
        self.bn4 = torch.nn.BatchNorm3d(out_channels)
        self.relu = torch.nn.ReLU()

        self.identity_downsample = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
            torch.nn.BatchNorm3d(out_channels)
        )

    def forward(self, x):
        idx = x.clone()
        idx = self.identity_downsample(idx)
        x = self.conv3_1(x)
       
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv3_2(x)
        x = self.bn2(x)
        x += idx
        x = self.relu(x)
        idx2 = x.clone()
        x = self.conv3_3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv3_4(x)
        x = self.bn4(x)
        x += idx2
        x = self.relu(x)
        return x

class Res183d(torch.nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super(Res183d, self).__init__()
        self.conv1 = Conv1(in_channels=1, out_channels=64)
        self.conv2 = Conv2_x(in_channels=64, out_channels=64)
        self.conv3 = Conv3_x(in_channels=64, out_channels=128)
        self.conv4 = Conv4_x(in_channels=128, out_channels=256)
        self.conv5 = Conv5_x(in_channels=256, out_channels=512)
        self.apool = torch.nn.AdaptiveAvgPool3d((1,1,1))
        self.fc = torch.nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.apool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
