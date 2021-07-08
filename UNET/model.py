import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class Section(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Section, self).__init__()
        self.process = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.process(x)

class UNET(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super(UNET, self).__init__()
        # Contraction
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down1 = Section(in_channels=in_channels, out_channels=64, kernel_size=3)
        self.down2 = Section(in_channels=64, out_channels=128, kernel_size=3)
        self.down3 = Section(in_channels=128, out_channels=256, kernel_size=3)
        self.down4 = Section(in_channels=256, out_channels=512, kernel_size=3)
        self.down5 = Section(in_channels=512, out_channels=1024, kernel_size=3)
        # Expansion
        self.up_conv1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.up_conv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up_conv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up_conv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.up1 = Section(in_channels=1024, out_channels=512, kernel_size=3)
        self.up2 = Section(in_channels=512, out_channels=256, kernel_size=3)
        self.up3 = Section(in_channels=256, out_channels=128, kernel_size=3)
        self.up4 = Section(in_channels=128, out_channels=64, kernel_size=3)
        self.output = self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        skip_connections = []
        # CONTRACTION
        # down 1
        x = self.down1(x)
        skip_connections.append(x)
        x = self.pool(x)
        # down 2
        x = self.down2(x)
        skip_connections.append(x)
        x = self.pool(x)
        # down 3
        x = self.down3(x)
        skip_connections.append(x)
        x = self.pool(x)
        # down 4
        x = self.down4(x)
        skip_connections.append(x)
        x = self.pool(x)
        # down 5
        x = self.down5(x)
        
        # EXPANSION
        # up1
        x = self.up_conv1(x)
        y = skip_connections[3]
        y = TF.resize(y, size=[56, 56])
        y_new = torch.cat((y, x), dim=1)
        x = self.up1(y_new)
        
        # up2
        x = self.up_conv2(x)
        y = skip_connections[2]
        # resize skip commention
        y = TF.resize(y, size=[104, 104])
        y_new = torch.cat((y, x), dim=1)
        x = self.up2(y_new)
        # up3
        x = self.up_conv3(x)
        y = skip_connections[1]
        y = TF.resize(y, size=[200, 200])
        y_new = torch.cat((y, x), dim=1)
        x = self.up3(y_new)
        # up4 
        x = self.up_conv4(x)
        y = skip_connections[0]
        y = TF.resize(y, size=[392, 392])
        y_new = torch.cat((y, x), dim=1)
        x = self.up4(y_new)
        x = self.output(x)
        return x
        
model = UNET()