"""整体流程的基础配置与路径常量。

包含数据根目录、类别数量、训练超参数及默认的结果输出目录，供各模块统一引用。
"""

from pathlib import Path

# 项目路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent

# 数据与类别配置
DATA_ROOT = REPO_ROOT / "data1" / "data_noise_30"
KNOWN_CLASS_COUNT = 7
UNKNOWN_CLASS_COUNT = 3
TOTAL_CLASSES = KNOWN_CLASS_COUNT + UNKNOWN_CLASS_COUNT

# 训练超参数
RECON_BATCH_SIZE = 256
SUPERVISED_CONTRASTIVE_WEIGHT = 0.20
SUPERVISED_CONTRASTIVE_TEMPERATURE = 0.2
SUPERVISED_CE_WEIGHT = 0.0
SUPERVISED_EPOCHS = 15
SUPERVISED_BATCH_SIZE = 128
SUPERVISED_LR = 1e-4
FREEZE_BACKBONE_IN_SUP = False

# 输出目录
RECON_DIR = REPO_ROOT / "result" / "MIX" / "reconstruction_outputs_30"
RECON_DIR.mkdir(parents=True, exist_ok=True)
OPEN_SET_DIR = REPO_ROOT / "result" / "MIX" / "true_open_set_results_data1_30"
OPEN_SET_DIR.mkdir(parents=True, exist_ok=True)
