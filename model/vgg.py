import torch
from torch import nn
import torch.nn.functional as F 

class VggNet(nn.Module):
    def __init__(self, in_channels=1, conv_arch=((1, 64), (1, 128), (2, 256), (2, 512), (2, 512)), input_size=224):
        super(VggNet, self).__init__()
        num_blocks = len(conv_arch)
        minst_map_size = int(input_size/(2 ** num_blocks))
        blocks = []
        for i, (num_convs, out_channels) in enumerate(conv_arch):
            blocks.append(self.block(in_channels, out_channels, num_convs))
            in_channels = out_channels
        self.vgg_blocks = nn.Sequential(*blocks)
        # 全连接层部分
        self.fc = nn.Sequential(nn.Flatten(),
                               nn.Linear(out_channels * minst_map_size * minst_map_size, 4096),
                               nn.ReLU(),
                               nn.Dropout(0.5),
                               nn.Linear(4096, 4096),
                               nn.ReLU(),
                               nn.Dropout(0.5),
                               nn.Linear(4096, 10))

        
    def forward(self, x):
        x = self.vgg_blocks(x)
        x = self.fc(x)
        return x
    
    def block(self,in_channels, out_channels, num_convs):
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
        return nn.Sequential(*layers)
