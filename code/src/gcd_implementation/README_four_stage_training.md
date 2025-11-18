# 四阶段训练流程说明

## 概述

在原有的三阶段GCD训练流程基础上，我们新增了第四阶段：**未知类别检测与微调**。这个阶段专门用于提升模型对未知类别的识别能力。

## 训练流程

### 阶段1：自监督预训练
- 使用所有数据进行无监督重构学习
- 训练backbone提取有效特征
- 训练重构头进行信号重构

### 阶段2：有标签监督微调
- 只使用有标签数据进行监督学习
- 优化分类头在已知类别上的性能
- 冻结或微调backbone参数

### 阶段3：初步评估
- 在完整数据集上评估当前模型性能
- 输出详细的分类结果分布

### 阶段4：未知类别检测与微调 (新增)
- **检测阶段**：使用置信度和熵阈值从无标签数据中分离未知样本
- **微调阶段**：使用特殊的分布损失函数在未知样本上进行微调
- **最终评估**：评估未知类别微调后的整体性能

## 核心组件

### 1. UnknownClassDetector
未知类别检测器，负责：
- 分析模型预测的置信度和熵
- 根据阈值识别未知样本
- 创建专门的未知类别数据集

**关键参数**：
- `confidence_threshold`: 置信度阈值 (默认: 0.6)
- `entropy_threshold`: 熵阈值 (默认: 1.5)
- `use_entropy_threshold`: 是否使用熵阈值

### 2. UnknownClassFinetuner
未知类别微调器，负责：
- 实现特殊的未知类别分布损失函数
- 让模型更好地识别和区分未知类别
- 平衡分布集中度和多样性

**关键参数**：
- `entropy_weight`: 熵正则化权重 (默认: 0.1)
- `distribution_sharpness`: 分布锐化参数 (默认: 5.0)
- `freeze_backbone_unknown`: 是否在未知微调中冻结backbone

### 3. 损失函数设计

#### 未知类别分布损失
```python
def unknown_distribution_loss(self, logits: torch.Tensor) -> torch.Tensor:
    # 1. Mask已知类别
    masked_logits = logits.clone()
    masked_logits[:, :self.known_class_count] = -10.0

    # 2. 找到未知类别中的最大logits
    unknown_max_logits, unknown_max_indices = torch.max(masked_logits, dim=1)

    # 3. 创建目标分布：只在最大未知类别上有高概率
    target_distribution = torch.zeros_like(logits)
    sharp_probs = F.softmax(unknown_max_logits * self.distribution_sharpness, dim=0)
    target_distribution.scatter_(1, unknown_max_indices.unsqueeze(1), sharp_probs.unsqueeze(1))

    # 4. 计算KL散度损失
    current_probs = F.softmax(logits, dim=1)
    loss = F.kl_div(current_probs.log(), target_distribution, reduction='batchmean')

    return loss
```

#### 熵正则化
```python
def entropy_regularization(self, logits: torch.Tensor) -> torch.Tensor:
    # 只计算未知类别的熵
    masked_logits = logits.clone()
    masked_logits[:, :self.known_class_count] = -10.0

    unknown_probs = F.softmax(masked_logits, dim=1)
    entropy = -torch.sum(unknown_probs * torch.log(unknown_probs + 1e-8), dim=1)

    # 期望熵 = log(未知类别数量)
    target_entropy = torch.log(torch.tensor(self.num_total_classes - self.known_class_count, dtype=torch.float32))
    entropy_loss = (entropy - target_entropy).pow(2).mean()

    return entropy_loss
```

## 使用方法

### 基本命令
```bash
python code/gcd_implementation/train_gcd.py \
    --arch_type ca1d \
    --data_root data/LFM_dataset \
    --known_class_count 7 \
    --unknown_class_count 3 \
    --epochs_pretrain 15 \
    --epochs_finetune 30 \
    --epochs_unknown_finetune 15
```

### 未知类别微调参数
```bash
python code/gcd_implementation/train_gcd.py \
    [原有参数...] \
    --confidence_threshold 0.6 \
    --entropy_threshold 1.5 \
    --lr_unknown_finetune 5e-4 \
    --entropy_weight 0.1 \
    --distribution_sharpness 5.0 \
    --freeze_backbone_unknown
```

## 输出文件

训练完成后会生成以下文件：
- `pretrain_backbone_{arch_type}.pth`: 预训练的backbone
- `finetuned_gcd_model_{arch_type}.pth`: 有标签微调后的模型
- `unknown_indices_{arch_type}.pth`: 检测到的未知样本索引
- `final_gcd_model_with_unknown_{arch_type}.pth`: 四阶段训练后的最终模型

## 预期效果

1. **更好的未知类别识别**：模型能更准确地区分已知和未知类别
2. **集中的未知预测分布**：未知样本的预测更加集中和自信
3. **提升开集识别性能**：整体的开集识别能力得到增强

## 测试验证

运行测试脚本验证功能：
```bash
python test_four_stage_training.py
```

该测试会验证：
- 未知类别检测器的正常工作
- 未知类别微调器的训练流程
- 损失函数的计算正确性
- 评估指标的输出

## 注意事项

1. **置信度阈值**：如果模型训练较好，可能需要降低置信度阈值以检测更多未知样本
2. **学习率**：未知类别微调通常使用较小的学习率
3. **训练轮数**：未知类别微调的轮数一般少于有标签微调
4. **熵权重**：过大的熵权重可能导致分布过于分散

## 故障排除

### 问题：未检测到未知样本
**解决方案**：
- 降低置信度阈值（如从0.6降到0.4）
- 检查模型是否在已知类别上过拟合
- 确认无标签数据中确实包含未知类别样本

### 问题：未知样本预测过于分散
**解决方案**：
- 增加 `distribution_sharpness` 参数
- 减小 `entropy_weight` 参数
- 增加训练轮数

### 问题：训练损失为负值
**这是正常现象**：
- KL散度损失可能为负值，这表示目标分布比当前分布更加集中
- 负损失不表示训练错误，而是表示模型正在向目标分布收敛