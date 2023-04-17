import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils import *


# resnet bottleneck
class BottleNeck(nn.Module):
    def __init__(self, in_channel, hidden_channel):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=hidden_channel, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=hidden_channel, out_channels=hidden_channel, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels=hidden_channel, out_channels=in_channel, kernel_size=1)
        self.activate = nn.ReLU()
        self.batchNorm1 = nn.BatchNorm2d(hidden_channel)
        self.batchNorm2 = nn.BatchNorm2d(in_channel)
        self.dropout = nn.Dropout(0.5)
        init_param(self.modules())

    def forward(self, x):
        x1 = self.activate(self.conv1(x))
        x2 = self.activate(self.conv2(x1))
        x3 = self.activate(self.conv3(x2))
        return x + x3


# resnet basic block
class BasicBlock(nn.Module):
    def __init__(self, channels):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, stride=1)
        self.activate = nn.ReLU()
        self.batchNorm = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout(0.1)
        init_param(self.modules())

    def forward(self, x):
        x1 = self.activate(self.conv1(x))
        # x1 = self.activate(self.batchNorm(self.conv1(x)))
        x2 = self.activate(self.conv2(x1))
        # x2 = self.activate(self.batchNorm(self.conv2(x1)))
        # x3 = self.conv3(x2)
        return x + x2


# SEnet basic block
class SEBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(SEBlock, self).__init__()
        self.squeeze = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1)
        self.exitation = nn.Conv2d(in_channels=hidden_channels, out_channels=in_channels, kernel_size=1)
        self.activate = nn.ReLU()
        init_param(self.modules())

    def forward(self, x):
        x1 = F.adaptive_avg_pool2d(x, (1, 1))
        x2 = self.activate(self.activate(self.squeeze(x1)))
        x3 = self.exitation(x2)
        return torch.sigmoid(x3)


class SEBasicBlock(nn.Module):
    def __init__(self, channels):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, stride=1)
        self.SEBlock = SEBlock(in_channels=channels, hidden_channels=int(channels / 16))
        self.activate = nn.ReLU()
        init_param(self.modules())

    def forward(self, x):
        se = self.SEBlock(x)
        x1 = self.activate(self.conv1(x))
        x2 = self.activate(self.conv2(x1))
        x3 = x2 * se
        return x + x3


# resnet bottleneck
class SEBottleNeck(nn.Module):
    def __init__(self, in_channel, hidden_channel):
        super(SEBottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=hidden_channel, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=hidden_channel, out_channels=hidden_channel, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels=hidden_channel, out_channels=in_channel, kernel_size=1)
        self.SEBlock = SEBlock(in_channels=in_channel, hidden_channels=int(in_channel / 16))
        self.activate = nn.ReLU()
        self.batchNorm = nn.BatchNorm2d(in_channel)
        init_param(self.modules())

    def forward(self, x):
        se = self.SEBlock(x)
        x1 = self.activate(self.conv1(x))
        x2 = self.activate(self.conv2(x1))
        x3 = self.batchNorm(self.activate(self.conv3(x2)))
        x4 = x3 * se
        return x + x4