# dataset.py
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


class ImageSet(Dataset):
    """
    数据类：按80%训练、20%测试分割
    """

    def __init__(self, list_path, args=None, train=False):
        self.train = train
        self.data = []
        self.labels = []
        for root, dirs, files in os.walk(list_path):
            for label, name in enumerate(files):
                data_path = os.path.join(list_path, name)
                mat_file = sio.loadmat(data_path)
                img_data = mat_file['all_data']
                img_data = np.squeeze(img_data)
                self.data.append(img_data)
                data_label = np.full(img_data.shape[0], label)
                self.labels.append(data_label)
        self.data = torch.from_numpy(
            np.array(np.concatenate(self.data))).float()
        self.labels = torch.from_numpy(
            np.array(np.concatenate(self.labels))).long()
        self.train_data = []
        self.train_labels = []
        self.test_data = []
        self.test_labels = []
        for i in range(len(set(self.labels))):
            index = (self.labels == i)
            index_imgs = self.data[index]
            index_len = int(len(index_imgs) * 0.8)  # 直接80%训练、20%测试
            self.train_data.append(self.data[index][:index_len])
            self.test_data.append(self.data[index][index_len:])
            self.train_labels.append(self.labels[index][:index_len])
            self.test_labels.append(self.labels[index][index_len:])
        self.train_data = torch.cat(self.train_data, dim=0)
        self.test_data = torch.cat(self.test_data, dim=0)
        self.train_labels = torch.cat(self.train_labels, dim=0)
        self.test_labels = torch.cat(self.test_labels, dim=0)

    def __getitem__(self, index):
        if self.train:
            img, lb = self.train_data[index], self.train_labels[index]
        else:
            img, lb = self.test_data[index], self.test_labels[index]
        return img, lb

    def __len__(self):
        if self.train:
            return len(self.train_labels)
        else:
            return len(self.test_labels)
