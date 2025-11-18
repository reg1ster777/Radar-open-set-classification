import torch
import torch.nn as nn

_MAMBA_AVAILABLE = True
_IMPORT_ERROR = None
try:
    from mamba_ssm import Mamba
except ImportError as exc:  # pragma: no cover - only triggered without dependency
    _MAMBA_AVAILABLE = False
    _IMPORT_ERROR = exc


class MambaBackbone(nn.Module):
    """
    简单封装的 Mamba 主干网络。

    该实现假设输入为一维雷达信号 [B, C, L]，首先通过 1x1 卷积映射到指定的隐藏维度，
    随后串联多层 Mamba 块，最后使用 LayerNorm 与自适应池化获得固定长度的特征向量。
    """

    def __init__(
        self,
        input_channels: int = 1,
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if not _MAMBA_AVAILABLE:
            raise ImportError(
                "MambaBackbone 需要额外依赖 `mamba-ssm`，请先安装: pip install mamba-ssm"
            ) from _IMPORT_ERROR

        self.input_proj = nn.Conv1d(input_channels, d_model, kernel_size=1)
        self.sequence_layers = nn.ModuleList(
            [
                nn.Sequential(
                    Mamba(
                        d_model=d_model,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand,
                    ),
                    nn.Dropout(p=dropout),
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, L] 输入信号
        Returns:
            features: [B, d_model] 特征向量
        """
        x = self.input_proj(x)            # [B, d_model, L]
        x = x.transpose(1, 2)             # [B, L, d_model]

        for layer in self.sequence_layers:
            x = layer(x)

        x = self.norm(x)
        x = x.transpose(1, 2)             # [B, d_model, L]
        x = self.pool(x).squeeze(-1)      # [B, d_model]
        return x


__all__ = ["MambaBackbone"]
