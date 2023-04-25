"""
    Implement patch-based Face image Super-Resolution(FSR) problem.

    The resolution of high-resolution(HR) images is 120*100. The low-resolution(LR) images are obtained by
    performing down-sampling(by a factor of 4) and smoothing(by a 4*4 averaging filter) on HR images,
    therefore the resolution of LR images is 30*25.

    The first 320 images in FEI face database are used as basic images for linear combination,
    therefore the search space of this problem is 320-D. Decision vector should be an array of length 320,
    while the elements are supposed to be in the range [-1, 1] and the sum of elements should be 1.
    The next 40 images are used as input images of training set, and last 40 images are used as
    input images of test set.

    Facial images are respectively cropped into 594(27*22) patches by default, and it depends on the
    parameters "patch_size" and "overlap"(both in LR space) you passed to the class "FSR_Dataset".
    Formulas of computing the number of patches are as follows:
        number_of_patches_vertically = ceil((30 - overlap) / (patch_size - overlap))
        number_of_patches_horizontally = ceil((25 - overlap) / (patch_size - overlap)
        number_of_patches = number_of_patches_vertically * number_of_patches_horizontally
"""
import numpy as np
from torch.utils.data import Dataset
import os
from os import path
import cv2

from L2OBench.Problem import Basic_Problem


class FSR(Basic_Problem):
    def __init__(self, image_id, input_patches_LR, input_img_LR, input_img_HR, basic_patches_LR, basic_patches_HR,
                 scale=4, patch_size_LR=4, overlap_LR=3):
        self.image_id = image_id
        self.input_patches_LR = input_patches_LR  # [n_patches, patch_size_LR * patch_size_LR]
        self.input_img_LR = input_img_LR          # [30, 25]
        self.input_img_HR = input_img_HR          # [120, 100]
        self.basic_patches_LR = basic_patches_LR  # [n_patches, 320, patch_size_LR * patch_size_LR]
        self.basic_patches_HR = basic_patches_HR  # [n_patches, 320, patch_size_HR * patch_size_HR]
        self.scale = scale
        self.patch_size_LR = patch_size_LR
        self.overlap_LR = overlap_LR
        self.optimum = 0.

    def eval(self, x: np.ndarray):
        """
        :param x: should be a 3-D array of shape [number_of_patches, NP, 320].
        :return: The costs in a 2-D array of shape [number_of_patches, NP].
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if x.ndim == 3:
            return self.func(x)
        else:
            raise ValueError(f'Parameter "x" should have 3 dimensions, but the provided has {x.ndim} dimension(s).')

    def func(self, x: np.ndarray):
        """
        :param x: should be a 3-D array of shape [number_of_patches, NP, 320].
        :return: The costs in a 2-D array of shape [number_of_patches, NP].
        """
        lin_comb = np.matmul(x, self.basic_patches_LR)   # linear combination of basic patches [n_patches, NP, patch_size_LR * patch_size_LR]
        return np.sum(np.power(np.expand_dims(self.input_patches_LR, 1) - lin_comb, 2), axis=-1)  # least square  [n_patches, NP]

    def output_result(self, save_path: str, x: np.ndarray):  # x.shape=[n_patches, 320]
        """
        :param save_path: the path to save images.
        :param x: should be a 2-D array of shape [number_of_patches, 320].
        :return: Outputs three .jpg images in save path.
        """
        if not path.exists(save_path):
            os.makedirs(save_path)
        cv2.imwrite(path.join(save_path, f'{self.image_id}_LR.jpg'), self.input_img_LR)
        cv2.imwrite(path.join(save_path, f'{self.image_id}_HR.jpg'), self.input_img_HR)
        # compute the result image
        height_HR = self.input_img_LR.shape[0] * self.scale
        width_HR = self.input_img_LR.shape[1] * self.scale
        patch_size_HR = self.patch_size_LR * self.scale
        overlap_HR = self.overlap_LR * self.scale
        n_patch_ver = int(np.ceil((self.input_img_LR.shape[0] - self.overlap_LR) / (self.patch_size_LR - self.overlap_LR)))  # number of patches vertically
        n_patch_hor = int(np.ceil((self.input_img_LR.shape[1] - self.overlap_LR) / (self.patch_size_LR - self.overlap_LR)))   # number of patches horizontally
        result = np.zeros((height_HR, width_HR))
        counter = np.zeros((height_HR, width_HR))
        lin_comb = np.squeeze(np.matmul(np.expand_dims(x, 1), self.basic_patches_HR))  # [n_patch, patch_size_HR * patch_size_HR]
        for i in range(n_patch_ver):
            start_row = min(i * self.scale * (self.patch_size_LR - self.overlap_LR),
                            height_HR - self.scale * self.patch_size_LR)
            for j in range(n_patch_hor):
                start_col = min(j * (patch_size_HR - overlap_HR),
                                width_HR - patch_size_HR)
                result[start_row: start_row + patch_size_HR,
                       start_col: start_col + patch_size_HR] += lin_comb[i * n_patch_hor + j].reshape(patch_size_HR, patch_size_HR)
                counter[start_row: start_row + patch_size_HR,
                        start_col: start_col + patch_size_HR] += 1
        result = result / counter
        cv2.imwrite(path.join(save_path, f'{self.image_id}_result.jpg'), result)


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
        n_patch_ver = int(np.ceil((height_LR - overlap) / (patch_size - overlap)))  # number of patches vertically
        n_patch_hor = int(np.ceil((width_LR - overlap) / (patch_size - overlap)))   # number of patches horizontally
        kernel = np.ones((scale, scale)) / (scale ** 2)  # averaging filter for smoothing

        # load basic images and crop them into patches
        basic_patches_LR = [[] for _ in range(n_patch_ver * n_patch_hor)]
        basic_patches_HR = [[] for _ in range(n_patch_ver * n_patch_hor)]
        for i in range(0, 160):
            for j in ('a', 'b'):
                HR = cv2.cvtColor(cv2.imread(path.join(data_folder, f'{i + 1}{j}.jpg')), cv2.COLOR_BGR2GRAY)
                LR = cv2.filter2D(cv2.resize(HR, (width_LR, height_LR)), -1, kernel)
                for m in range(n_patch_ver):
                    start_row = min(m * (patch_size - overlap), height_LR - patch_size)
                    for n in range(n_patch_hor):
                        start_col = min(n * (patch_size - overlap), width_LR - patch_size)
                        basic_patch_LR = LR[start_row: start_row + patch_size,
                                            start_col: start_col + patch_size]
                        basic_patch_HR = HR[start_row * scale: (start_row + patch_size) * scale,
                                            start_col * scale: (start_col + patch_size) * scale]
                        basic_patches_LR[m * n_patch_hor + n].append(basic_patch_LR.reshape(-1))
                        basic_patches_HR[m * n_patch_hor + n].append(basic_patch_HR.reshape(-1))
        for patch in range(n_patch_ver * n_patch_hor):
            basic_patches_LR[patch] = np.stack(basic_patches_LR[patch])
            basic_patches_HR[patch] = np.stack(basic_patches_HR[patch])
        basic_patches_LR = np.stack(basic_patches_LR)
        basic_patches_HR = np.stack(basic_patches_HR)

        # load input images and crop them into patches
        data = []
        start_image_idx = 160 if not test else 180
        for i in range(start_image_idx, start_image_idx + 20):
            for j in ('a', 'b'):
                HR = cv2.cvtColor(cv2.imread(path.join(data_folder, f'{i + 1}{j}.jpg')), cv2.COLOR_BGR2GRAY)
                LR = cv2.filter2D(cv2.resize(HR, (width_LR, height_LR)), -1, kernel)
                input_patches = []
                for m in range(n_patch_ver):
                    start_row = min(m * (patch_size - overlap), height_LR - patch_size)
                    for n in range(n_patch_hor):
                        start_col = min(n * (patch_size - overlap), width_LR - patch_size)
                        input_patch = LR[start_row: start_row + patch_size,
                                         start_col: start_col + patch_size]
                        input_patches.append(input_patch.reshape(-1))
                input_patches = np.stack(input_patches)
                data.append(FSR(f'{i + 1}{j}',
                                input_patches,
                                LR,
                                HR,
                                basic_patches_LR,
                                basic_patches_HR,
                                scale,
                                patch_size,
                                overlap))
        return data
