# ===============================
# File: model_common.py (refactored: remove D, single classifier C)
# ===============================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import itertools
import os
import os.path as osp
from datetime import datetime


class BaseSolver:
    """
    求解器：特征提取器 G 和单分类器 C。
    """

    def __init__(self, num_classes=7, lr=1e-9, device="cpu", center_loss_weight=0.003):
        """
        初始化BaseSolver

        Args:
            num_classes (int): 分类类别数
            lr (float): 学习率
            device (str): 计算设备
            center_loss_weight (float): 中心损失权重
        """
        self.device = device
        self.num_k = num_classes
        self.lr = lr
        self.center_loss_weight = center_loss_weight
        self.model_type = None

        # 核心组件（由子类实现）
        self.G = None  # 特征提取器
        self.C = None  # 分类器
        self.centers = None  # 形状 [num_k, feat_dim]，每个类别的中心特征

        # 损失函数与优化器
        self.optimizer = None
        self.xent_loss = nn.CrossEntropyLoss().to(device)  # 交叉熵损失函数

    def setup_optimizer(self):
        """设置优化器，在组件初始化后调用"""
        assert self.G is not None and self.C is not None and self.centers is not None
        # 将特征提取器、分类器和中心特征的参数合并到一个参数迭代器中
        all_params = itertools.chain(
            self.G.parameters(), self.C.parameters(), [self.centers]
        )
        self.optimizer = optim.Adam(all_params, lr=self.lr)

    def reset_grad(self):
        """清空优化器的梯度"""
        self.optimizer.zero_grad()

    def compute_center_loss(self, features, labels):
        """
        计算中心损失

        Args:
            features (Tensor): 特征向量 [B, feat_dim]
            labels (Tensor): 标签 [B]

        Returns:
            Tensor: 中心损失值
        """
        # 根据标签选择对应的类别中心
        centers_batch = self.centers.index_select(0, labels.long())
        # 计算特征与对应类别中心的均方误差
        return F.mse_loss(features, centers_batch)

    def train(self, source_loader, test_loader, lr, epochs, remarks, test=False):
        """
        训练模型

        Args:
            source_loader: 训练数据加载器
            test_loader: 测试数据加载器
            epochs (int): 训练轮数
            remarks (str): 模型保存备注
            test (bool): 是否在训练过程中进行测试，默认为False
        """
        print(f"=== 模型架构: {self.model_type} ===")
        print(f"=== 中心损失权重: {self.center_loss_weight} ===")

        self.lr = lr
        self.set_lr(self.lr)
        print(f"=== 学习率: {self.lr} ===")
        

        for epoch in range(epochs):
            self.G.train()
            self.C.train()

            total_loss = 0.0
            total_xent_loss = 0.0
            total_center_loss = 0.0

            for batch in source_loader:
                # 获取图像和标签数据
                img = batch[0].to(self.device).unsqueeze(1)  # 增加通道维度
                label = batch[1].to(self.device)

                self.reset_grad()

                # 前向传播
                feat = self.G(img)              # [B, feat_dim] 提取特征
                logits = self.C(feat)           # [B, num_k] 分类预测

                # 计算损失
                loss_xent = self.xent_loss(logits, label)  # 交叉熵损失
                loss_center = self.compute_center_loss(feat, label)  # 中心损失
                loss = loss_xent + self.center_loss_weight * loss_center  # 总损失

                # 反向传播和优化
                loss.backward()
                self.optimizer.step()

                # 累计损失
                total_loss += loss.item()
                total_xent_loss += loss_xent.item()
                total_center_loss += loss_center.item()

            # 计算平均损失
            avg_total_loss = total_loss / len(source_loader)
            avg_xent_loss = total_xent_loss / len(source_loader)
            avg_center_loss = total_center_loss / len(source_loader)

            print(
                f">>> Epoch {epoch+1}/{epochs}: Avg Total Loss = {avg_total_loss:.6f} | "
                f"Avg Xent Loss = {avg_xent_loss:.6f} | "
                f"Avg Center Loss = {avg_center_loss:.6f}"
            )

            # 如果需要测试，则在测试集上评估
            if test:
                self.test_epoch(test_loader, epoch+1)

        # 保存模型
        self.save_model(remarks=remarks)
        print("=== 训练结束 ===")

    def test_epoch(self, test_loader, epoch=None):
        """
        测试集评估：整体准确率 + 各类别准确率

        Args:
            test_loader: 测试数据加载器
            epoch (int): 当前训练轮数，可选
        """
        self.G.eval()
        self.C.eval()

        class_correct = {}  # 每个类别的正确预测数
        class_total = {}    # 每个类别的总样本数

        with torch.no_grad():  # 不计算梯度
            for batch in test_loader:
                # 获取图像和标签数据
                img = batch[0].to(self.device).unsqueeze(1)
                label = batch[1].to(self.device)

                # 前向传播
                feat = self.G(img)
                logits = self.C(feat)
                pred_label = logits.argmax(dim=1)  # 预测标签

                # 统计每个类别的正确预测数和总样本数
                for i in range(len(label)):
                    lb = label[i].item()
                    pred_lb = pred_label[i].item()
                    if lb not in class_total:
                        class_total[lb] = 0
                        class_correct[lb] = 0
                    class_total[lb] += 1
                    if pred_lb == lb:
                        class_correct[lb] += 1

        # 计算整体准确率
        total_correct = sum(class_correct.values())
        total_samples = sum(class_total.values())
        overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        print(f"Overall Accuracy: {overall_accuracy * 100:.2f}%")
        # 输出每个类别的准确率
        for label in sorted(class_total.keys()):
            acc = class_correct[label] / class_total[label]
            print(
                f"  Class {label}: {acc * 100:.2f}% ({class_correct[label]}/{class_total[label]})"
            )
        print("=" * 30)

        # 恢复训练模式
        self.G.train()
        self.C.train()

    def save_model(self, remarks):
        """
        保存模型参数到指定路径（仅 G、C、centers、optimizer）

        Args:
            remarks (str): 模型保存备注
        """
        save_path = f"model/{self.model_type}_class{self.num_k}"
        if not osp.exists(save_path):
            os.makedirs(save_path)

        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = osp.join(save_path, f"{remarks}_{timestamp}.pth")

        # 保存模型状态
        torch.save({
            'G_state_dict': self.G.state_dict(),
            'C_state_dict': self.C.state_dict(),
            'centers': self.centers,
            'optimizer': self.optimizer.state_dict(),
        }, model_path)
        print(f"{self.model_type} 参数已保存至: {model_path}")

    def load_model(self, checkpoint_path):
        """
        从指定路径加载模型参数

        Args:
            checkpoint_path (str): 模型检查点路径
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.G.load_state_dict(checkpoint['G_state_dict'])
        self.C.load_state_dict(checkpoint['C_state_dict'])

        # 加载中心特征
        if 'centers' in checkpoint:
            self.centers.data.copy_(checkpoint['centers'])
        # 加载优化器状态
        if 'optimizer' in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        # 设置为训练模式
        self.G.train()
        self.C.train()
        print(f"=== {self.model_type} 参数已成功加载 ===")

    def forward_feature(self, x):
        """
        仅返回 G 的特征，用于外部评估/可视化。

        Args:
            x (Tensor): 输入数据

        Returns:
            Tensor: 特征向量
        """
        return self.G(x)
