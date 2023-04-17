import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils import *
from model.blocks import *

class BasicBlockModel(nn.Module):
    def __init__(self, cls_nums):
        super(BasicBlockModel, self).__init__()
        cov_channels = [1, 8, 16, 8]
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=cov_channels[1], kernel_size=1)
        self.block1 = BasicBlock(channels=cov_channels[1])
        # self.block1 = BottleNeck(in_channel=cov_channels[1], hidden_channel=int(cov_channels[1] / 4))

        self.conv2 = nn.Conv2d(in_channels=cov_channels[1], out_channels=cov_channels[2], kernel_size=1)
        self.block2 = BasicBlock(channels=cov_channels[2])
        # self.block2 = BottleNeck(in_channel=cov_channels[2], hidden_channel=int(cov_channels[2] / 4))

        self.conv3 = nn.Conv2d(in_channels=cov_channels[2], out_channels=cov_channels[3], kernel_size=1)
        self.block3 = BasicBlock(channels=cov_channels[3])
        # self.block3 = BottleNeck(in_channel=cov_channels[3], hidden_channel=int(cov_channels[3] / 4))

        fc_channels = [392, 64, cls_nums] # 这里的第一个值，如果调整了输入尺寸，需要随之调整 [72, 32, cls_nums] [392, 128, cls_nums]
        self.fc1 = nn.Linear(in_features=fc_channels[0], out_features=fc_channels[1])
        self.fc2 = nn.Linear(in_features=fc_channels[1], out_features=fc_channels[2])

        self.activate = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.batchNorm1 = nn.BatchNorm2d(cov_channels[1])
        self.batchNorm2 = nn.BatchNorm2d(cov_channels[2])
        self.batchNorm3 = nn.BatchNorm2d(cov_channels[3])
        self.batchNorm4 = nn.BatchNorm1d(fc_channels[1])
        self.final_activate = nn.ReLU()
        self.pooling_kernel = (2, 2)
        init_param(self.modules())

    def forward(self, x:torch.Tensor):
        # x = self.activate(self.conv1(x))
        # x = F.max_pool2d(self.activate(self.conv1(x)), kernel_size=self.pooling_kernel)
        x = F.max_pool2d(self.activate(self.batchNorm1(self.conv1(x))), kernel_size=self.pooling_kernel)
        x = F.max_pool2d(self.block1(x), kernel_size=self.pooling_kernel)

        # x = self.activate(self.conv2(x))
        # x = self.activate(self.batchNorm2(self.conv2(x)))
        x = F.max_pool2d(self.activate(self.batchNorm2(self.conv2(x))), kernel_size=self.pooling_kernel)
        x = F.max_pool2d(self.block2(x), kernel_size=self.pooling_kernel)

        # x = self.activate(self.conv3(x))
        x = self.activate(self.batchNorm3(self.conv3(x)))
        # x = F.max_pool2d(self.activate(self.batchNorm3(self.conv3(x))), kernel_size=self.pooling_kernel)
        x = F.max_pool2d(self.block3(x), kernel_size=self.pooling_kernel)

        x = x.reshape(-1, self.fc1.in_features)
        # x = self.activate(self.dropout(self.fc1(x)))
        # x = self.activate(self.dropout(self.fc2(x)))
        x = self.activate(self.dropout(self.fc1(x)))
        x = self.fc2(x)
        return x

class BottleNeckModel(nn.Module):
    def __init__(self, cls_nums):
        super(BottleNeckModel, self).__init__()
        cov_channels = [1, 32, 64, 32]
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=cov_channels[1], kernel_size=1)
        self.block1 = BottleNeck(in_channel=cov_channels[1], hidden_channel=int(cov_channels[1] / 4))
        self.conv2 = nn.Conv2d(in_channels=cov_channels[1], out_channels=cov_channels[2], kernel_size=1)
        self.block2 = BottleNeck(in_channel=cov_channels[2], hidden_channel=int(cov_channels[2] / 4))

        fc_channels = [12544, 1024, 64, cls_nums]  # 这里的第一个值，如果调整了输入尺寸，需要随之调整
        self.fc1 = nn.Linear(in_features=fc_channels[0], out_features=fc_channels[1])
        self.fc2 = nn.Linear(in_features=fc_channels[1], out_features=fc_channels[2])
        self.fc3 = nn.Linear(in_features=fc_channels[2], out_features=fc_channels[3])
        self.activate = nn.ReLU()

        init_param(self.modules())

    def forward(self, x:torch.Tensor):
        x = F.avg_pool2d(self.activate(self.conv1(x)), kernel_size=(2, 2))
        x = F.avg_pool2d(self.block1(x), kernel_size=(2, 2))
        x = F.avg_pool2d(self.activate(self.conv2(x)), kernel_size=(2, 2))
        x = F.avg_pool2d(self.block2(x), kernel_size=(2, 2))

        x = x.reshape(-1, self.fc1.in_features)
        x = self.activate(self.fc1(x))
        x = self.activate(self.fc2(x))
        x = self.activate(self.fc3(x))
        return x


