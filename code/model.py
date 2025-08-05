# model.py
# @Time: 2025/08/05
# @Author: reg1ster

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import os
import os.path as osp
from dataset import ImageSet


class ChannelAttention(nn.Module):
    """
    通道注意力机制
    """

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        avg_out: 全局平均池化 -> 1D卷积层 -> 激活函数 -> 1D卷积层
        max_out: 全局最大池化 -> 1D卷积层 -> 激活函数 -> 1D卷积层

        return: avg_out + max_out
        """
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class Feature_base(nn.Module):
    """
    G: 特征生成器
    """

    def __init__(self):
        super(Feature_base, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu1 = nn.ReLU()
        self.ca1 = ChannelAttention(32)
        self.conv2 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=5, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu2 = nn.ReLU()
        self.ca2 = ChannelAttention(64)
        self.conv3 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=7, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu3 = nn.ReLU()
        self.ca3 = ChannelAttention(128)
        self.conv4 = nn.Conv1d(
            in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.ca1(x) * x

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.ca2(x) * x

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.ca3(x) * x

        x = self.gap(x)
        x = self.flatten(x)
        return x


class Feature_disentangle(nn.Module):
    """
    D_di/D_ci: 特征解耦器
    """

    def __init__(self):
        super(Feature_disentangle, self).__init__()
        self.fc1 = nn.Linear(128, 96)
        self.bn1_fc = nn.BatchNorm1d(96)
        self.fc2 = nn.Linear(96, 64)
        self.bn2_fc = nn.BatchNorm1d(64)

    def forward(self, x):
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn2_fc(self.fc2(x)))
        return x


class Predictor(nn.Module):
    """
    C_di/C_ci: 分类器
    """

    # Adjust output_dim to num_classes
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=7, prob=0.3):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(prob)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x
