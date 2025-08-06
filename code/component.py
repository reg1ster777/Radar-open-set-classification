# component.py
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
import math
from dataset import ImageSet

# === SimplifiedSolver 简化模型 ===
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


class G_FeatureBase(nn.Module):
    """
    G: 特征生成器
    """

    def __init__(self):
        super(G_FeatureBase, self).__init__()
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


class D_FeatureDisentangle(nn.Module):
    """
    D_di/D_ci: 特征解耦器
    """

    def __init__(self):
        super(D_FeatureDisentangle, self).__init__()
        self.fc1 = nn.Linear(128, 96)
        self.bn1_fc = nn.BatchNorm1d(96)
        self.fc2 = nn.Linear(96, 64)
        self.bn2_fc = nn.BatchNorm1d(64)

    def forward(self, x):
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn2_fc(self.fc2(x)))
        return x


class C_Predictor(nn.Module):
    """
    C_di/C_ci: 分类器
    """

    # Adjust output_dim to num_classes
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=7, prob=0.3):
        super(C_Predictor, self).__init__()
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


# === HybridSolver 混合模型 CNN+transformer ===
class PositionalEncoding(nn.Module):
    """
    位置编码层
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # 形状: [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class G_HybridFeatureBase(nn.Module):
    """
    混合特征提取器（CNN浅层 + Transformer深层）
    """

    def __init__(self):
        super(G_HybridFeatureBase, self).__init__()
        # --- 浅层CNN（局部特征提取） ---
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

        # --- 深层Transformer（全局依赖建模） ---
        self.pos_encoder = PositionalEncoding(d_model=64)  # 输入维度=64
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=8,       # 8个注意力头
            dim_feedforward=256,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=2)

        # --- 输出层 ---
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        # 浅层CNN
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.ca1(x) * x  # 形状: [batch, 32, L1]

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.ca2(x) * x  # 形状: [batch, 64, L2]

        # 准备Transformer输入: [seq_len, batch, features]
        x = x.permute(2, 0, 1)  # [L2, batch, 64]
        x = self.pos_encoder(x)  # 添加位置编码

        # Transformer处理
        x = self.transformer_encoder(x)  # 输出: [seq_len, batch, 64]

        # 全局平均池化
        x = x.permute(1, 2, 0)  # [batch, 64, seq_len]
        x = self.gap(x)         # [batch, 64, 1]
        x = self.flatten(x)      # [batch, 64]
        return x


class D_HybridFeatureDisentangle(nn.Module):
    """
    特征解耦器（调整输入维度为64）
    """

    def __init__(self, input_dim=64):  # 修改：支持动态输入维度
        super(D_HybridFeatureDisentangle, self).__init__()
        self.fc1 = nn.Linear(input_dim, 96)
        self.bn1_fc = nn.BatchNorm1d(96)
        self.fc2 = nn.Linear(96, 64)
        self.bn2_fc = nn.BatchNorm1d(64)

    def forward(self, x):
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn2_fc(self.fc2(x)))
        return x
