# solver.py
"""
统一的架构选择器（仅保留此函数）：
  - "ca1d" → model_ca1d_cnn.CA1DCNNSolver
  - "hybrid" → model_hybrid_cnn_tf.HybridCNNTFSolver
"""


def solver_choose(arch_type, num_classes=7, lr=1e-4, device="cpu", center_loss_weight=0.003):
    """
    arch_type:  "hybrid"     混合架构求解器
                "ca1d"       普通架构求解器
    """
    common_params = {
        "num_classes": num_classes,
        "lr": lr,
        "device": device,
        "center_loss_weight": center_loss_weight
    }
    name = str(arch_type).lower()
    if name == "hybrid" or name == "cnn_tf" or name == "hybrid_cnn_tf":
        from .model_hybrid_cnn_tf import HybridCNNTFSolver
        return HybridCNNTFSolver(**common_params)
    elif name == "ca1d" or name == "ca1d_cnn":
        from .model_ca1d_cnn import CA1DCNNSolver
        return CA1DCNNSolver(**common_params)
    else:
        raise ValueError(f"无效架构类型: '{arch_type}'")
