import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn

# Model
def down_conv(in_channels, out_channels, kernel_size, padding, stride):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
    return block

def up_conv(in_channels, out_channels, kernel_size, padding, stride, output_padding):
    block = nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, output_padding=output_padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
    return block

class Skip_AE(nn.Module):
    def __init__(self):
        super(Skip_AE, self).__init__()

        self.down_conv1 = down_conv(in_channels = 1, out_channels = 64, kernel_size = 5, padding = 2, stride = 1)
        self.down_conv2 = down_conv(in_channels = 64, out_channels = 128, kernel_size = 5, padding = 2, stride = 2)
        self.down_conv3 = down_conv(in_channels = 128, out_channels = 256, kernel_size = 5, padding = 2, stride = 2)
        self.down_conv4 = down_conv(in_channels = 256, out_channels = 512, kernel_size = 5, padding = 2, stride = 2)
        self.down_conv5 = down_conv(in_channels = 512, out_channels = 64, kernel_size = 5, padding = 2, stride = 2)
        
        self.up_conv1 = up_conv(in_channels=64, out_channels=512, kernel_size=5, stride=2, padding=2, output_padding = 1)
        self.up_conv2 = up_conv(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=2, output_padding = 1)
        self.up_conv3 = up_conv(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding = 1)
        self.up_conv4 = up_conv(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding = 1)
        self.up_conv5 = up_conv(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, output_padding=0)
        
    def forward(self, image):
        #encoder
        x1 = self.down_conv1(image)
        x2 = self.down_conv2(x1)
        x3 = self.down_conv3(x2)
        x4 = self.down_conv4(x3)
        x5 = self.down_conv5(x4)
        
        #decoder
        y1 = self.up_conv1(x5)
        y1 = y1 + x4
        y2 = self.up_conv2(y1)
        y2 = nn.Dropout(0.1)(y2 + x3)
        y3 = self.up_conv3(y2)
        y3 = nn.Dropout(0.1)(y3 + x2)
        y4 = self.up_conv4(y3)
        y4 = nn.Dropout(0.1)(y4 + x1)
        y5 = self.up_conv5(y4)
        out = torch.tanh(y5)
        return out