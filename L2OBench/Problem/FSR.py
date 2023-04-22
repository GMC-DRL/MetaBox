"""
Implement patch-based Face image Super-Resolution(FSR) problem.
"""

import numpy as np
from torch.utils.data import Dataset
from os import path
import cv2

from L2OBench.Problem import Basic_Problem


class FSR(Basic_Problem):
    def __init__(self, input_patch, train_patch):
        self.input_patch = input_patch  # [patch_size * patch_size]
        self.train_patch = train_patch  # [n_train_img, patch_size * patch_size]
        self.optimum = 0.

    def func(self, x):  # x.shape=[NP, n_train_img]
        lin_comb = np.matmul(x, self.train_patch)   # linear combination of training batches
        return np.sum(np.power(self.input_patch - lin_comb, 2), axis=-1)  # least square


class FSR_Dataset(Dataset):
    height_HR = 120
    width_HR = 100

    def __init__(self,
                 batch_size=1,
                 patch_size=4,   # size of patch in LR space
                 overlap=3,      # num of overlap pixels in LR space
                 test=False
                 ):
        self.batch_size = batch_size
        self.data = FSR_Dataset.load_data(4, patch_size, overlap, test)
        self.N = len(self.data)
        # initialize pointer for iteratively getting data batch
        self.ptr = [i for i in range(0, self.N, batch_size)]
        # initialize the order data being selected, preparation for shuffling
        self.index = np.arange(self.N)

    def __getitem__(self, item):
        ptr = self.ptr[item]
        index = self.index[ptr: min(ptr + self.batch_size, self.N)]
        res = []
        for i in range(len(index)):
            res.append(self.data[index[i]])
        return res

    def __len__(self):
        return self.N

    def shuffle(self):
        self.index = np.random.permutation(self.N)

    @classmethod
    def load_data(cls, scale=4, patch_size=4, overlap=3, test=False):
        data_folder = path.join(path.dirname(__file__), 'FSR_data')
        height_LR, width_LR = cls.height_HR // scale, cls.width_HR // scale
        n_patch_ver = int(np.ceil((height_LR - overlap) / (patch_size - overlap)))
        n_patch_hor = int(np.ceil((width_LR - overlap) / (patch_size - overlap)))
        kernel = np.ones((scale, scale)) / (scale ** 2)
        train_patch = [[] for _ in range(n_patch_ver * n_patch_hor)]

        for i in range(0, 160):
            for j in ('a', 'b'):
                HR = cv2.cvtColor(cv2.imread(path.join(data_folder, f'{i + 1}{j}.jpg')), cv2.COLOR_BGR2GRAY)
                LR = cv2.filter2D(cv2.resize(HR, (width_LR, height_LR)), -1, kernel)
                for m in range(n_patch_ver):
                    start_row = min(m * (patch_size - overlap), height_LR - patch_size)
                    for n in range(n_patch_hor):
                        start_col = min(n * (patch_size - overlap), width_LR - patch_size)
                        patch = LR[start_row: start_row + patch_size, start_col: start_col + patch_size]
                        train_patch[m * n_patch_hor + n].append(patch.reshape(-1))
        for patch in range(len(train_patch)):
            train_patch[patch] = np.vstack(train_patch[patch])

        data = []
        start_image = 160 if not test else 180
        for i in range(start_image, start_image + 20):
            for j in ('a', 'b'):
                HR = cv2.cvtColor(cv2.imread(path.join(data_folder, f'{i + 1}{j}.jpg')), cv2.COLOR_BGR2GRAY)
                LR = cv2.filter2D(cv2.resize(HR, (width_LR, height_LR)), -1, kernel)
                for m in range(n_patch_ver):
                    start_row = min(m * (patch_size - overlap), height_LR - patch_size)
                    for n in range(n_patch_hor):
                        start_col = min(n * (patch_size - overlap), width_LR - patch_size)
                        input_patch = LR[start_row: start_row + patch_size, start_col: start_col + patch_size]
                        data.append(FSR(input_patch.reshape(-1), train_patch[m * n_patch_hor + n]))
        return data
