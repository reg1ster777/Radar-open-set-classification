# ===============================
# File: model_hybrid_cnn_tf.py (refactored: remove D, single C; feat_dim=64)
# ===============================
from .model_common import BaseSolver
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class G_HybridFeatureBase(nn.Module):
    def __init__(self):
        super().__init__()
        # 浅层CNN
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu1 = nn.ReLU()
        self.ca1 = ChannelAttention(32)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu2 = nn.ReLU()
        self.ca2 = ChannelAttention(64)

        # Transformer
        self.pos_encoder = PositionalEncoding(d_model=64)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=8, dim_feedforward=256, dropout=0.1, batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=2)

        # 输出层
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        # CNN
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.ca1(x) * x
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.ca2(x) * x

        # [B,64,L] -> [L,B,64]
        x = x.permute(2, 0, 1)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)  # [L,B,64]

        # GAP
        x = x.permute(1, 2, 0)  # [B,64,L]
        x = self.gap(x)
        x = self.flatten(x)     # [B,64]
        return x


class C_Predictor(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=7, prob=0.3):
        super().__init__()
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


class HybridCNNTFSolver(BaseSolver):
    def __init__(self, num_classes=7, lr=1e-9, device="cpu", center_loss_weight=0.003):
        super().__init__(num_classes, lr, device, center_loss_weight)
        self.model_type = "Hybrid-CNN-TF"

        # 初始化组件（仅 G 和 C）
        self.G = G_HybridFeatureBase().to(self.device)
        self.C = C_Predictor(
            input_dim=64, output_dim=self.num_k).to(self.device)
        self.centers = nn.Parameter(
            torch.randn(self.num_k, 64).to(self.device))

        # 设置优化器
        self.setup_optimizer()

from typing import Union


class HybridCNNTFSolver(BaseSolver):
    def __init__(self, num_classes=7, lr=1e-9, device="cpu", center_loss_weight=0.003):
        super().__init__(num_classes, lr, device, center_loss_weight)
        self.model_type = "Hybrid-CNN-TF"

        # 初始化组件（仅 G 和 C）
        self.G = G_HybridFeatureBase().to(self.device)
        self.C = C_Predictor(
            input_dim=64, output_dim=self.num_k).to(self.device)
        self.centers = nn.Parameter(
            torch.randn(self.num_k, 64).to(self.device))

        # 设置优化器
        self.setup_optimizer()

    # 阶段前调用，确保实际生效的 lr 与传参一致
    def set_lr(self, lr: float):
        self.lr = float(lr)
        if getattr(self, "optimizer", None) is None:
            self.setup_optimizer()
            return
        for g in self.optimizer.param_groups:
            g["lr"] = self.lr

    # （可选）若需要重建优化器（比如换了参数组），可以加这个：
    def reconfigure_optimizer(self, lr: Union[float, None] = None):
        if lr is not None:
            self.lr = float(lr)
        self.setup_optimizer()

