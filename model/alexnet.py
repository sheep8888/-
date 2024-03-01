import torch
from torch import nn
import torch.nn.functional as F 

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv_block1 = nn.Sequential(nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=3, stride=2))
        # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
        self.conv_block2 = nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, padding=2),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=3, stride=2))
        # 使用三个连续的卷积层和较小的卷积窗口。
        # 除了最后的卷积层，输出通道的数量进一步增加。
        # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
        self.conv_block3 = nn.Sequential(nn.Conv2d(256, 384, kernel_size=3, padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(384, 384, kernel_size=3, padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(384, 256, kernel_size=3, padding=1),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=3, stride=2))
        self.fc = nn.Sequential(nn.Flatten(),
                                nn.Linear(6400, 4096),
                                nn.ReLU(),
                                nn.Dropout(p=0.5),
                                nn.Linear(4096, 4096),
                                nn.ReLU(),
                                nn.Dropout(p=0.5),
                                nn.Linear(4096, 10))

        
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.fc(x)
        return x
