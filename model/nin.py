import torch
from torch import nn


class NiNNet(nn.Module):
    def __init__(self, in_channels=1):
        super(NiNNet, self).__init__()
        self.nin_blocks = nn.Sequential(
            self.block(in_channels, 96, kernel_size=11, strides=4, padding=0),
            nn.MaxPool2d(3, stride=2),
            self.block(96, 256, kernel_size=5, strides=1, padding=2),
            nn.MaxPool2d(3, stride=2),
            self.block(256, 384, kernel_size=3, strides=1, padding=1),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(0.5),
            # 标签类别数是10
            self.block(384, 10, kernel_size=3, strides=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            # 将四维的输出转成二维的输出，其形状为(批量大小,10)
            nn.Flatten()
        )

        
    def forward(self, x):
        x = self.nin_blocks(x)
        return x
    
    def block(self,in_channels, out_channels, kernel_size, strides, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
            nn.BatchNorm2d(out_channels),nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),nn.ReLU())
