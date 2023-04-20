import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils import *

class Model(nn.Module):
    def __init__(self, cls_nums):
        super(Model, self).__init__()
        cov_channels = [3, 32, 64, 32]
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=cov_channels[0], kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=cov_channels[0], out_channels=cov_channels[1], kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels=cov_channels[1], out_channels=cov_channels[2], kernel_size=3, padding=1, stride=1)
        self.conv4 = nn.Conv2d(in_channels=cov_channels[2], out_channels=cov_channels[3], kernel_size=3, padding=1, stride=1)

        fc_channels = [6272, 1024, 64, cls_nums] # 这里的第一个值，如果调整了输入尺寸，需要随之调整
        self.fc1 = nn.Linear(in_features=fc_channels[0], out_features=fc_channels[1])
        self.fc2 = nn.Linear(in_features=fc_channels[1], out_features=fc_channels[2])
        self.fc3 = nn.Linear(in_features=fc_channels[2], out_features=fc_channels[3])
        self.activate = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        init_param(self.modules())

    def forward(self, x:torch.Tensor):
        x = F.avg_pool2d(self.activate(self.conv1(x)), kernel_size=(2, 2))
        x = F.avg_pool2d(self.activate(self.conv2(x)), kernel_size=(2, 2))
        x = F.avg_pool2d(self.activate(self.conv3(x)), kernel_size=(2, 2))
        x = F.avg_pool2d(self.activate(self.conv4(x)), kernel_size=(2, 2))
        x = x.reshape(-1, self.fc1.in_features)
        x = self.activate(self.fc1(x))
        x = self.activate(self.fc2(x))
        x = self.activate(self.fc3(x))
        return x


