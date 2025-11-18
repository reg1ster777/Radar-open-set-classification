# ===============================
# File: model_ca1d_cnn.py (refactored: remove D, single C; feat_dim=128)
# ===============================
from .model_common import BaseSolver
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class G_FeatureBase(nn.Module):
    def __init__(self):
        super().__init__()
        # block1
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.ca1 = ChannelAttention(32)

        # block2
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.ca2 = ChannelAttention(64)

        # block3
        self.conv3 = nn.Conv1d(64, 128, kernel_size=7, padding=3)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.ca3 = ChannelAttention(128)

        # block4（膨胀卷积扩感受野）
        self.conv4 = nn.Conv1d(128, 128, kernel_size=5, padding=4, dilation=2)
        self.bn4 = nn.BatchNorm1d(128)
        self.relu4 = nn.ReLU()
        self.ca4 = ChannelAttention(128)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x); x = self.relu1(x); x = self.pool1(x)
        x = self.ca1(x) * x
        x = self.conv2(x)
        x = self.bn2(x); x = self.relu2(x); x = self.pool2(x)
        x = self.ca2(x) * x
        x = self.conv3(x)
        x = self.bn3(x); x = self.relu3(x); x = self.pool3(x)
        x = self.ca3(x) * x
        x = self.conv4(x)
        x = self.bn4(x); x = self.relu4(x)
        x = self.ca4(x) * x
        x = self.gap(x)
        x = self.flatten(x)  # [B,128]
        return x


class C_Predictor(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, output_dim=7, prob=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)   # ← 替换 BN1 为 LN1
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)   # ← 替换 BN2 为 LN2
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(prob)

    def forward(self, x):
        # x: [B, input_dim]
        x = self.fc1(x)
        x = F.relu(self.ln1(x))   # 先线性 -> LN -> ReLU
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(self.ln2(x))
        x = self.fc3(x)
        return x



class CA1DCNNSolver(BaseSolver):
    def __init__(self, num_classes=7, lr=1e-9, device="cpu", center_loss_weight=0.003):
        super().__init__(num_classes, lr, device, center_loss_weight)
        self.model_type = "CA-1D-CNN"

        # 初始化组件（仅 G 和 C）
        self.G = G_FeatureBase().to(self.device)
        self.C = C_Predictor(input_dim=128, output_dim=self.num_k).to(self.device)
        self.centers = nn.Parameter(torch.randn(self.num_k, 128).to(self.device))

        # 设置优化器
        self.setup_optimizer()

     # 在阶段切换/调用训练前，显式同步学习率
    def set_lr(self, lr: float):
        """将新的 lr 同步到优化器的所有 param_groups，并更新 self.lr"""
        self.lr = float(lr)
        if getattr(self, "optimizer", None) is None:
            # 还没有优化器就按当前 lr 建一个
            self.setup_optimizer()
            return
        for g in self.optimizer.param_groups:
            g["lr"] = self.lr

    # （可选）若需要重建优化器（比如换了参数组），可以加这个：
    def reconfigure_optimizer(self, lr: float | None = None):
        """可选：按当前（或给定）lr 重新创建优化器"""
        if lr is not None:
            self.lr = float(lr)
        self.setup_optimizer()
