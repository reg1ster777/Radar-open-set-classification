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
from model import Feature_base, Feature_disentangle, Predictor
from datetime import datetime


class SimplifiedSolver:
    def __init__(self, num_classes=7, lr=0.0000001, device="cpu"):
        self.device = device
        self.num_k = num_classes
        self.G = Feature_base().to(self.device)
        self.D_di = Feature_disentangle().to(self.device)
        self.D_ci = Feature_disentangle().to(self.device)
        self.C_di = Predictor(output_dim=num_classes).to(self.device)
        self.C_ci = Predictor(output_dim=num_classes).to(self.device)
        self.opt_G = optim.Adam(self.G.parameters(), lr=lr)
        self.opt_D_di = optim.Adam(self.D_di.parameters(), lr=lr)
        self.opt_D_ci = optim.Adam(self.D_ci.parameters(), lr=lr)
        self.opt_C_di = optim.Adam(self.C_di.parameters(), lr=lr)
        self.opt_C_ci = optim.Adam(self.C_ci.parameters(), lr=lr)
        self.xent_loss = nn.CrossEntropyLoss().to(self.device)

    def reset_grad(self):
        """
        重置梯度
        """
        self.opt_G.zero_grad()
        self.opt_D_di.zero_grad()
        self.opt_D_ci.zero_grad()
        self.opt_C_di.zero_grad()
        self.opt_C_ci.zero_grad()

    def optimize_classifier(self, pred_di, pred_ci, label):
        """
        分类交叉熵损失
        """
        loss_di = self.xent_loss(pred_di, label)
        loss_ci = self.xent_loss(pred_ci, label)
        return loss_di + loss_ci

    def train(self, source_loader, test_loader, epochs, test=False):
        """
        训练过程
        """
        print("=== 模型架构: SimplifiedSolver ===")
        for epoch in range(epochs):
            total_loss = 0
            for batch in source_loader:
                img = batch[0].to(self.device).unsqueeze(1)
                label = batch[1].to(self.device)

                # 1. --- reset_grad ---
                self.reset_grad()

                # 2. --- feature ---
                feat_src = self.G(img)              # 基础特征提取 [batch, C, L]
                feat_di = self.D_di(feat_src)       # 解耦域不变特征
                feat_ci = self.D_ci(feat_src)       # 解耦域相关特征
                pred_di = self.C_di(feat_di)        # 基于域不变特征的分类预测
                pred_ci = self.C_ci(feat_ci)        # 基于域相关特征的分类预测

                # 3. --- loss ---
                class_loss = self.optimize_classifier(pred_di, pred_ci, label)
                # TODO: ring_loss = self.ring_loss_minimizer(feat_src)

                # 4. --- backward ---
                class_loss.backward()               # 计算梯度
                self.opt_G.step()                   # 更新特征提取器
                self.opt_D_di.step()                # 更新域不变解耦器
                self.opt_D_ci.step()                # 更新域相关解耦器
                self.opt_C_di.step()                # 更新域不变分类器
                self.opt_C_ci.step()                # 更新域相关分类器
                total_loss += class_loss.item()

            avg_loss = total_loss / len(source_loader)
            print(f">>> Epoch {epoch+1}/{epochs}: Avg Loss = {avg_loss:.6f}")
            if test == True:
                self.test_epoch(test_loader, epoch+1)
        
        self.save_model()
        print("=== 训练结束 ===")

    def test_epoch(self, test_loader, epoch=None):
        """
        测试函数
        """
        # --- 切换到评估模式 ---
        self.G.eval()
        self.D_di.eval()
        self.D_ci.eval()
        self.C_di.eval()
        self.C_ci.eval()

        # --- 初始化类别统计字典 ---
        class_correct = {}
        class_total = {}

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                img = batch[0].to(self.device).unsqueeze(1)
                label = batch[1].to(self.device)

                # --- feature ---
                feat_src = self.G(img)              # 基础特征提取 [batch, C, L]
                feat_di = self.D_di(feat_src)       # 解耦域不变特征
                pred_di = self.C_di(feat_di)        # 基于域不变特征的分类预测
                pred_label = pred_di.argmax(dim=1)  # 取概率最高的类别

                # --- analyse ---
                for i in range(len(label)):
                    lb = label[i].item()
                    preb_lb = pred_label[i].item()

                    if lb not in class_total:
                        class_total[lb] = 0
                        class_correct[lb] = 0

                    class_total[lb] += 1
                    if preb_lb == lb:
                        class_correct[lb] += 1

        total_correct = sum(class_correct.values())
        total_samples = sum(class_total.values())
        overall_accuracy = total_correct / total_samples

        # --- print ---
        print(f"Overall Accuracy: {overall_accuracy * 100:.2f}%")
        for label in sorted(class_total.keys()):
            acc = class_correct[label] / class_total[label]
            print(
                f"  Class {label}: {acc * 100:.2f}% ({class_correct[label]}/{class_total[label]})")
        print("=" * 30)

        # --- 恢复训练模式 ---
        self.G.train()
        self.D_di.train()
        self.D_ci.train()
        self.C_di.train()
        self.C_ci.train()

    def save_model(self):
        """保存参数"""
        save_path = "model"
        if not osp.exists(save_path):
            os.makedirs(save_path)

        now_time = datetime.now()
        timestamp = now_time.strftime("%Y%m%d_%H%M%S")
        model_path = osp.join(save_path, f"model_{timestamp}.pth")

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
        print(f"模型参数已保存至: {model_path}")


    # def test(self, test_loader):
    #     self.G.eval()
    #     self.D_di.eval()
    #     self.C_di.eval()
    #     correct = 0
    #     total = 0
    #     with torch.no_grad():
    #         for batch_idx, batch in enumerate(test_loader):
    #             img = batch[0].to(self.device).unsqueeze(1)
    #             lb = batch[1].to(self.device)
    #             feat = self.G(img)
    #             feat_di = self.D_di(feat)
    #             pred_logits = self.C_di(feat_di)
    #             pred_lb = pred_logits.argmax(dim=1)
    #             # 打印分类结果和真实标签（完整tensor，如果太长可截取如 pred_lb[:256]）
    #             print(f"分类结果：pred_lb {pred_lb}")
    #             print(f"真实标签：lb {lb}")
    #             correct += (pred_lb == lb).sum().item()
    #             total += lb.size(0)
    #     accuracy = correct / total
    #     print(f"Test Accuracy: {accuracy * 100:.2f}%")
