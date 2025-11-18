import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入现有的Backbone模型
from src.solver.model_ca1d_cnn import G_FeatureBase as CA1DCNNBackbone
from src.solver.model_hybrid_cnn_tf import G_HybridFeatureBase as HybridCNNTransformerBackbone
try:
    from src.gcd_implementation.mamba_backbone import MambaBackbone
except ImportError:  # pragma: no cover - optional dependency
    MambaBackbone = None


class ReconstructionHead(nn.Module):
    def __init__(self, feature_dim: int, original_signal_length: int, input_channels: int = 1, compression_ratio: float = 0.25):
        super().__init__()
        # 添加压缩瓶颈：feature_dim -> compressed_dim -> feature_dim
        compressed_dim = int(feature_dim * compression_ratio)

        # 压缩编码器
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, compressed_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 重构解码器：从压缩特征重构完整信号
        self.decoder = nn.Sequential(
            nn.Linear(compressed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, original_signal_length * input_channels)
        )

        self.original_signal_length = original_signal_length
        self.input_channels = input_channels
        self.compressed_dim = compressed_dim

    def forward(self, x):
        # x: [batch_size, feature_dim]

        # 压缩特征
        compressed_features = self.encoder(x)  # [batch_size, compressed_dim]

        # 重构完整信号
        reconstructed_flat = self.decoder(compressed_features)

        # 重塑回信号的形状
        reconstructed_signal = reconstructed_flat.view(x.size(0), self.input_channels, self.original_signal_length)

        return reconstructed_signal, compressed_features


class LabeledReconstructionHead(nn.Module):
    """专门用于重构有标签数据的重构头"""
    def __init__(self, feature_dim: int, original_signal_length: int, input_channels: int = 1):
        super().__init__()
        # 比通用重构头更简单，专注于学习已知类模式
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, original_signal_length * input_channels)
        )
        self.original_signal_length = original_signal_length
        self.input_channels = input_channels

    def forward(self, x):
        # x: [batch_size, feature_dim]
        reconstructed_flat = self.decoder(x)
        reconstructed_signal = reconstructed_flat.view(x.size(0), self.input_channels, self.original_signal_length)
        return reconstructed_signal

    def compute_reconstruction_error(self, original_signal, reconstructed_signal):
        """计算重构误差，用于判断是否为已知类"""
        mse_error = F.mse_loss(reconstructed_signal, original_signal, reduction='none')
        # 沿信号维度平均，得到每个样本的重构误差
        return mse_error.mean(dim=(1, 2))  # [batch_size]


class ClassificationHead(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        # x: [batch_size, feature_dim]
        return self.linear(x)

    def forward_with_unknown_handling(self, x, reconstruction_errors, error_threshold, is_labeled_data=False):
        """
        简化版本：完全禁用未知检测，进行纯净的GCD训练

        Args:
            x: 特征 [batch_size, feature_dim]
            reconstruction_errors: 重构误差 [batch_size]
            error_threshold: 判断已知/未知类的阈值
            is_labeled_data: 是否为有标签数据
        """
        logits = self.linear(x)  # [batch_size, num_classes]

        # 完全禁用未知检测，所有数据都使用原始logits
        unknown_mask = torch.zeros_like(reconstruction_errors, dtype=torch.bool)  # 全部为已知
        return logits, unknown_mask


def get_backbone(arch_type: str, input_channels: int = 1):
    if arch_type == "ca1d":
        return CA1DCNNBackbone(), 128  # 返回Backbone实例和其输出特征维度
    elif arch_type == "hybrid":
        return HybridCNNTransformerBackbone(), 64 # 返回Backbone实例和其输出特征维度
    elif arch_type == "mamba":
        if MambaBackbone is None:
            raise ImportError(
                "未找到 MambaBackbone，可能缺少 `mamba-ssm` 依赖。请执行 `pip install mamba-ssm` 后重试。"
            )
        return MambaBackbone(input_channels=input_channels), 128
    else:
        raise ValueError(f"Unsupported architecture type: {arch_type}")


def get_labeled_reconstruction_head(feature_dim: int, original_signal_length: int, input_channels: int = 1):
    """获取有标签数据专用重构头"""
    return LabeledReconstructionHead(feature_dim, original_signal_length, input_channels)
