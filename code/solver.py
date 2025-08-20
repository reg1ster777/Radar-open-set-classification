# solver.py
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
from component import G_FeatureBase, D_FeatureDisentangle, C_Predictor
from component import G_HybridFeatureBase, D_HybridFeatureDisentangle
from datetime import datetime


def solver_choose(arch_type, num_classes=7, lr=0.0000001, device="cpu"):
    """
    arch_type:  "hybrid"    混合架构求解器
                "simplified"       普通架构求解器
    """
    if arch_type == "hybrid":
        solver = HybridSolver(
            num_classes=num_classes, lr=lr, device=device)
    elif arch_type == "simplified":
        solver = SimplifiedSolver(
            num_classes=num_classes, lr=lr, device=device)
    else:
        raise ValueError(f"无效架构类型: '{arch_type}'")
    return solver


class BaseSolver:
    """基础求解器类"""

    def __init__(self, num_classes=7, lr=0.0000001, device="cpu"):
        self.device = device
        self.num_k = num_classes
        self.lr = lr
        self._init_components()  # 初始化特定于子类的组件

        # 初始化优化器
        self.opt_G = optim.Adam(self.G.parameters(), lr=lr)
        self.opt_D_di = optim.Adam(self.D_di.parameters(), lr=lr)
        self.opt_D_ci = optim.Adam(self.D_ci.parameters(), lr=lr)
        self.opt_C_di = optim.Adam(self.C_di.parameters(), lr=lr)
        self.opt_C_ci = optim.Adam(self.C_ci.parameters(), lr=lr)

        self.xent_loss = nn.CrossEntropyLoss().to(self.device)

    def _init_components(self):
        """由子类重写以初始化特定组件"""
        raise NotImplementedError("子类必须实现此方法")

    def reset_grad(self):
        """重置梯度"""
        self.opt_G.zero_grad()
        self.opt_D_di.zero_grad()
        self.opt_D_ci.zero_grad()
        self.opt_C_di.zero_grad()
        self.opt_C_ci.zero_grad()

    def optimize_classifier(self, pred_di, pred_ci, label):
        """分类交叉熵损失"""
        loss_di = self.xent_loss(pred_di, label)
        loss_ci = self.xent_loss(pred_ci, label)
        return loss_di + loss_ci

    def train(self, source_loader, test_loader, epochs, test=False):
        """训练过程"""
        print(f"=== 模型架构: {self.model_type} ===")
        for epoch in range(epochs):
            total_loss = 0
            for batch in source_loader:
                img = batch[0].to(self.device).unsqueeze(1)
                label = batch[1].to(self.device)

                self.reset_grad()

                feat_src = self.G(img)
                feat_di = self.D_di(feat_src)
                feat_ci = self.D_ci(feat_src)
                pred_di = self.C_di(feat_di)
                pred_ci = self.C_ci(feat_ci)

                class_loss = self.optimize_classifier(pred_di, pred_ci, label)

                class_loss.backward()
                self.opt_G.step()
                self.opt_D_di.step()
                self.opt_D_ci.step()
                self.opt_C_di.step()
                self.opt_C_ci.step()

                total_loss += class_loss.item()

            avg_loss = total_loss / len(source_loader)
            print(f">>> Epoch {epoch+1}/{epochs}: Avg Loss = {avg_loss:.6f}")
            if test:
                self.test_epoch(test_loader, epoch+1)

        self.save_model()
        print("=== 训练结束 ===")

    def test_epoch(self, test_loader, epoch=None):
        """测试函数"""
        self.G.eval()
        self.D_di.eval()
        self.D_ci.eval()
        self.C_di.eval()
        self.C_ci.eval()

        class_correct = {}
        class_total = {}

        with torch.no_grad():
            for batch in test_loader:
                img = batch[0].to(self.device).unsqueeze(1)
                label = batch[1].to(self.device)

                feat_src = self.G(img)
                feat_di = self.D_di(feat_src)
                pred_di = self.C_di(feat_di)
                pred_label = pred_di.argmax(dim=1)

                for i in range(len(label)):
                    lb = label[i].item()
                    pred_lb = pred_label[i].item()
                    if lb not in class_total:
                        class_total[lb] = 0
                        class_correct[lb] = 0
                    class_total[lb] += 1
                    if pred_lb == lb:
                        class_correct[lb] += 1

        total_correct = sum(class_correct.values())
        total_samples = sum(class_total.values())
        overall_accuracy = total_correct / total_samples

        print(f"Overall Accuracy: {overall_accuracy * 100:.2f}%")
        for label in sorted(class_total.keys()):
            acc = class_correct[label] / class_total[label]
            print(
                f"  Class {label}: {acc * 100:.2f}% ({class_correct[label]}/{class_total[label]})")
        print("=" * 30)

        self.G.train()
        self.D_di.train()
        self.D_ci.train()
        self.C_di.train()
        self.C_ci.train()

    def save_model(self):
        """保存模型参数"""
        save_path = f"model/{self.model_type}"
        
        if not osp.exists(save_path):
            os.makedirs(save_path)

        now_time = datetime.now()
        timestamp = now_time.strftime("%Y%m%d_%H%M%S")
        model_path = osp.join(
            save_path, f"{self.model_type.lower()}_{timestamp}.pth")

        torch.save({
            'G_state_dict': self.G.state_dict(),
            'D_di_state_dict': self.D_di.state_dict(),
            'D_ci_state_dict': self.D_ci.state_dict(),
            'C_di_state_dict': self.C_di.state_dict(),
            'C_ci_state_dict': self.C_ci.state_dict(),
            'optimizer_G': self.opt_G.state_dict(),
            'optimizer_D_di': self.opt_D_di.state_dict(),
            'optimizer_D_ci': self.opt_D_ci.state_dict(),
            'optimizer_C_di': self.opt_C_di.state_dict(),
            'optimizer_C_ci': self.opt_C_ci.state_dict(),
        }, model_path)
        print(f"{self.model_type} 参数已保存至: {model_path}")

    def load_model(self, checkpoint_path):
        """加载预训练参数"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.G.load_state_dict(checkpoint['G_state_dict'])
        self.D_di.load_state_dict(checkpoint['D_di_state_dict'])
        self.D_ci.load_state_dict(checkpoint['D_ci_state_dict'])
        self.C_di.load_state_dict(checkpoint['C_di_state_dict'])
        self.C_ci.load_state_dict(checkpoint['C_ci_state_dict'])

        self.opt_G.load_state_dict(checkpoint['optimizer_G'])
        self.opt_D_di.load_state_dict(checkpoint['optimizer_D_di'])
        self.opt_D_ci.load_state_dict(checkpoint['optimizer_D_ci'])
        self.opt_C_di.load_state_dict(checkpoint['optimizer_C_di'])
        self.opt_C_ci.load_state_dict(checkpoint['optimizer_C_ci'])

        self.G.train()
        self.D_di.train()
        self.D_ci.train()
        self.C_di.train()
        self.C_ci.train()
        print(f"=== {self.model_type} 参数已成功加载 ===")


class SimplifiedSolver(BaseSolver):
    """普通架构求解器"""

    def __init__(self, num_classes=7, lr=0.0000001, device="cpu"):
        self.model_type = "SimplifiedSolver"
        super().__init__(num_classes, lr, device)

    def _init_components(self):
        """初始化普通架构的组件"""
        self.G = G_FeatureBase().to(self.device)
        self.D_di = D_FeatureDisentangle().to(self.device)
        self.D_ci = D_FeatureDisentangle().to(self.device)
        self.C_di = C_Predictor(output_dim=self.num_k).to(self.device)
        self.C_ci = C_Predictor(output_dim=self.num_k).to(self.device)


class HybridSolver(BaseSolver):
    """混合架构求解器（CNN-Transformer）"""

    def __init__(self, num_classes=7, lr=0.0000001, device="cpu"):
        self.model_type = "HybridSolver"
        super().__init__(num_classes, lr, device)

    def _init_components(self):
        """初始化混合架构的组件"""
        self.G = G_HybridFeatureBase().to(self.device)
        self.D_di = D_HybridFeatureDisentangle(input_dim=64).to(self.device)
        self.D_ci = D_HybridFeatureDisentangle(input_dim=64).to(self.device)
        self.C_di = C_Predictor(
            input_dim=64, output_dim=self.num_k).to(self.device)
        self.C_ci = C_Predictor(
            input_dim=64, output_dim=self.num_k).to(self.device)
