import torch
from torch import nn
import torch.nn.functional as F 

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv_block1 = nn.Sequential(nn.Conv2d(1, 6, kernel_size=5, padding=2),
                                   nn.Sigmoid(),
                                   nn.AvgPool2d(kernel_size=2, stride=2))
        self.conv_block2 = nn.Sequential(nn.Conv2d(1, 6, kernel_size=5, padding=2),
                                   nn.Sigmoid(),
                                   nn.AvgPool2d(kernel_size=2, stride=2))
        self.fc = nn.Sequential(nn.Flatten(),
                                nn.Linear(16 * 5 * 5, 120),
                                nn.Sigmoid(),
                                nn.Linear(120, 84),
                                nn.Sigmoid(),
                                nn.Linear(84, 10))
                                
        
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.fc(x)
        return x
