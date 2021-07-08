import torch
import torch.nn as nn

class Convolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(Convolution, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=kernel_size,
                             padding=padding,
                             stride=stride)
        self.Lrelu = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.Lrelu(x)
        return x

class YOLOv1(nn.Module):
    def __init__(self):
        super(YOLOv1, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Section 1
        # Tried all paddings from 0, 3 gives correct output shape
        self.section_1_conv = Convolution(in_channels=3, out_channels=64, kernel_size=7, padding=3, stride=2)
        # Section 2
        self.section_2_conv = Convolution(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding='same') #not strided conv
        # Section 3
        self.section_3_conv = nn.ModuleList([
            Convolution(in_channels=192, out_channels=128, kernel_size=1, stride=1, padding='same'),
            Convolution(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding='same'),
            Convolution(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding='same'),
            Convolution(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding='same')
        ])
        # section 4
        self.section_4_conv_1 = nn.ModuleList([
            Convolution(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding='same'),
            Convolution(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding='same')
        ])
        self.section_4_conv_2 = nn.ModuleList([
            Convolution(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding='same'),
            Convolution(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding='same')
        ])
        # section 5
        self.section_5_conv_1 = nn.ModuleList([
            Convolution(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding='same'),
            Convolution(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding='same')
        ])
        self.section_5_conv_2 = nn.ModuleList([
            Convolution(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding='same'),
            Convolution(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1)
        ])
        # section 6
        self.section_6_conv = Convolution(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding='same')
        # fc section
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024*7*7, 4096),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, 7*7*30),
            nn.LeakyReLU(0.1)
        )
        
    def forward(self, x):
        x = self.section_1_conv(x)
        x = self.pool(x)
        x = self.section_2_conv(x)
        x = self.pool(x)
        for sec_3 in self.section_3_conv:
            x = sec_3(x)
        x = self.pool(x)
        for i in range(0,4):
            for sec_4_1 in self.section_4_conv_1:
                x = sec_4_1(x)
        for sec_4 in self.section_4_conv_2:
            x = sec_4(x)
        x = self.pool(x)
        for i in range(0,2):
            for sec_5_1 in self.section_5_conv_1:
                x = sec_5_1(x)
        for sec_5 in self.section_5_conv_2:
            x = sec_5(x)        
        x = self.section_6_conv(x)
        x = self.section_6_conv(x)
        x = self.fc(x)
        x = torch.reshape(x, (7,7,30)) # reshape output
        return x