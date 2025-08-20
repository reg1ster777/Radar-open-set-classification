# train.py
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
from dataset import ImageSet
from solver import solver_choose


if __name__ == "__main__":
    # 路径
    known_class_path = 'data/data_noise_50/known_class_7'
    open_class_path = 'data/data_noise_50/open_class_10'

    # 加载训练集（80%）
    dataset_train = ImageSet(known_class_path, train=True)
    loader_train = DataLoader(dataset_train, batch_size=128, shuffle=True)
    print("=== 训练集加载完毕 ===")

    # 加载测试集（20%）
    dataset_test = ImageSet(known_class_path, train=False)
    loader_test = DataLoader(dataset_test, batch_size=128, shuffle=True)
    print("=== 测试集加载完毕 ===")

    # 获取设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== device: {} ===".format(device))

    # model_path = 'model\model_20250805_175559.pth'
    solver = solver_choose(arch_type="simplified", num_classes=7, device=device)
    # solver.load_model(model_path)  # 导入现有模型（可选）
    solver.train(loader_train, loader_test, epochs=500, test=True)
