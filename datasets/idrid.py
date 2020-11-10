#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/10/26 14:34
# @Author  : JackyLUO
# @E-mail  : lingluo@stumail.neu.edu.cn
# @Site    :
# @File    : idrid.py
# @Software: PyCharm

import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.utils.data as data

# Third party library
import albumentations as A
import cv2 as cv

from utils.visualizer import vis_keypoints
from utils.utils import gaussian_kernel

import matplotlib.pyplot as plt


class IDRID(data.Dataset):
    """
    REFUGE2018 data pipline
    """

    def __init__(self, cfg, mode='train'):
        mode = mode.lower()
        assert mode == 'train' or mode == 'test'

        self.sigma = cfg.MODEL.SIGMA
        self.label_type = cfg.MODEL.TARGET_TYPE
        self.stride = cfg.MODEL.STRIDE

        self.mode = mode
        self.output_size = cfg.MODEL.HEATMAP_SIZE
        self.path = cfg.DATASET.ROOT  # base path

        if self.mode == 'train':
            self.od = pd.read_csv(os.path.join(self.path, self.mode, 'groundtruths', cfg.DATASET.TRAINSET_OD))
            self.fovea = pd.read_csv(os.path.join(self.path, self.mode, 'groundtruths', cfg.DATASET.TRAINSET_FOVEA))
        elif self.mode == 'test':
            self.od = pd.read_csv(os.path.join(self.path, self.mode, 'groundtruths', cfg.DATASET.TESTSET_OD))
            self.fovea = pd.read_csv(os.path.join(self.path, self.mode, 'groundtruths', cfg.DATASET.TESTSET_FOVEA))

        od_num = len(self.od['Image No'].dropna(axis=0))  # remove NaN values
        fovea_num = len(self.fovea['Image No'].dropna(axis=0))
        assert od_num == fovea_num

        self.length = od_num
        self.parts = cfg.MODEL.NUM_JOINTS

    def __len__(self):
        return self.length

    def __data_pipline(self, img, ldmarks):
        # Convert RGB to BGR
        transform = None
        if self.mode == 'train':
            transform = A.Compose(
                [
                    A.Resize(height=self.output_size[0], width=self.output_size[1], p=1),  # /8--->(356, 536)
                    A.Crop(x_min=40, y_min=0, x_max=self.output_size[1] - 76, y_max=self.output_size[0], p=1),
                    # A.CLAHE(p=1),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.ToFloat(p=1),  # (0 ~ 1)
                    # A.Normalize(max_pixel_value=1, p=1)
                ],
                keypoint_params=A.KeypointParams(format='xy')
            )
        elif self.mode == 'test':
            # import random
            # random.seed(2020)
            transform = A.Compose(
                [
                    A.Resize(height=self.output_size[0], width=self.output_size[1], p=1),  # /8--->(356, 536)
                    A.Crop(x_min=40, y_min=0, x_max=self.output_size[1] - 76, y_max=self.output_size[0], p=1),
                    # (356, 460)
                    # A.CLAHE(p=1),
                    A.ToFloat(p=1),  # (0 ~ 1)
                    # A.Normalize(max_pixel_value=1, p=1)
                ],
                keypoint_params=A.KeypointParams(format='xy')
            )
        transformed = transform(image=img, keypoints=ldmarks)

        return transformed

    def __getitem__(self, idx):
        name = self.od.iloc[idx, 0]
        img = Image.open(os.path.join(self.path, self.mode, "images", name + ".jpg")).convert("RGB")

        od_x, od_y = self.od.iloc[idx, 1], self.od.iloc[idx, 2]
        fovea_x, fovea_y = self.fovea.iloc[idx, 1], self.fovea.iloc[idx, 2]

        keypoints = [
            (od_x, od_y),
            (fovea_x, fovea_y)
        ]

        transformed = self.__data_pipline(np.array(img), keypoints)
        t_img = transformed['image'].astype(np.float32)  # (356, 420, 3)
        H, W, _ = t_img.shape

        t_keypoints = transformed['keypoints']

        # vis_keypoints(t_img, t_keypoints)
        t_img = t_img.transpose([2, 0, 1])

        # Generate heatmaps
        heatmap = np.zeros((H // self.stride, W // self.stride, self.parts + 1), np.float32)  # (89, 105, 3)

        # Anatomical landmarks
        pts = np.array(t_keypoints, dtype=np.float32).reshape(-1, 2)
        tpts = np.array(t_keypoints, dtype=np.float32).reshape(-1, 2) / self.stride

        for i in range(self.parts):
            # Generate heatmaps
            kernel = gaussian_kernel(size_h=H // self.stride,
                                     size_w=W // self.stride,
                                     center_x=tpts[i][0], center_y=tpts[i][1],
                                     sigma=self.sigma)
            kernel[kernel > 1] = 1
            kernel[kernel < 0.01] = 0
            heatmap[:, :, i + 1] = kernel

        # Generate the heatmap of background
        heatmap[:, :, 0] = 1.0 - np.max(heatmap[:, :, 1:], axis=2)

        # plt.figure()
        # plt.subplot(221)
        # plt.imshow(np.transpose(np.uint8(t_img * 255), (1, 2, 0)))
        # plt.subplot(222)
        # plt.imshow(np.transpose(heatmap, (2, 0, 1))[0])
        # plt.subplot(223)
        # plt.imshow(np.transpose(heatmap, (2, 0, 1))[1])
        # plt.subplot(224)
        # plt.imshow(np.transpose(heatmap, (2, 0, 1))[2])
        # plt.show()

        image = torch.tensor(t_img, dtype=torch.float)
        heatmap = torch.tensor(np.transpose(heatmap, (2, 0, 1)), dtype=torch.float)
        pts = torch.tensor(pts, dtype=torch.float)
        # tpts = torch.tensor(tpts, dtype=torch.float)

        meta = {'index': idx, 'pts': pts}

        return image, heatmap, meta


class IDRIDFOVEA(data.Dataset):
    """
    REFUGE2018 data pipline (single point version)
    """

    def __init__(self, cfg, mode='train'):
        mode = mode.lower()
        assert mode == 'train' or mode == 'test'

        self.sigma = cfg.MODEL.SIGMA
        self.label_type = cfg.MODEL.TARGET_TYPE
        self.stride = cfg.MODEL.STRIDE

        self.mode = mode
        self.output_size = cfg.MODEL.HEATMAP_SIZE
        self.path = cfg.DATASET.ROOT  # base path

        if self.mode == 'train':
            self.fovea = pd.read_csv(os.path.join(self.path, self.mode, 'groundtruths', cfg.DATASET.TRAINSET_FOVEA))
        elif self.mode == 'test':
            self.fovea = pd.read_csv(os.path.join(self.path, self.mode, 'groundtruths', cfg.DATASET.TESTSET_FOVEA))

        fovea_num = len(self.fovea['Image No'].dropna(axis=0))

        self.length = fovea_num
        self.parts = cfg.MODEL.NUM_JOINTS

    def __len__(self):
        return self.length

    def __data_pipline(self, img, ldmarks):
        transform = None
        if self.mode == 'train':
            transform = A.Compose(
                [
                    A.Resize(height=self.output_size[0], width=self.output_size[1], p=1),
                    A.Crop(x_min=40, y_min=0, x_max=self.output_size[1] - 76, y_max=self.output_size[0], p=1),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.ToFloat(p=1),
                ],
                keypoint_params=A.KeypointParams(format='xy')
            )
        elif self.mode == 'test':
            transform = A.Compose(
                [
                    A.Resize(height=self.output_size[0], width=self.output_size[1], p=1),
                    A.Crop(x_min=40, y_min=0, x_max=self.output_size[1] - 76, y_max=self.output_size[0], p=1),
                    A.ToFloat(p=1),
                ],
                keypoint_params=A.KeypointParams(format='xy')
            )
        transformed = transform(image=img, keypoints=ldmarks)

        return transformed

    def __getitem__(self, idx):
        name = self.fovea.iloc[idx, 0]
        img = Image.open(os.path.join(self.path, self.mode, "images", name + ".jpg")).convert("RGB")

        fovea_x, fovea_y = self.fovea.iloc[idx, 1], self.fovea.iloc[idx, 2]

        keypoints = [
            (fovea_x, fovea_y)
        ]

        transformed = self.__data_pipline(np.array(img), keypoints)
        t_img = transformed['image'].astype(np.float32)
        H, W, _ = t_img.shape

        t_keypoints = transformed['keypoints']
        t_img = t_img.transpose([2, 0, 1])

        # Generate heatmaps
        heatmap = np.zeros((H // self.stride, W // self.stride, self.parts + 1), np.float32)

        # Anatomical landmarks
        pts = np.array(t_keypoints, dtype=np.float32).reshape(-1, 2)
        tpts = np.array(t_keypoints, dtype=np.float32).reshape(-1, 2) / self.stride

        for i in range(self.parts):
            kernel = gaussian_kernel(size_h=H // self.stride,
                                     size_w=W // self.stride,
                                     center_x=tpts[i][0], center_y=tpts[i][1],
                                     sigma=self.sigma)
            kernel[kernel > 1] = 1
            kernel[kernel < 0.01] = 0
            heatmap[:, :, i + 1] = kernel

        # Generate the heatmap of background
        heatmap[:, :, 0] = 1.0 - np.max(heatmap[:, :, 1:], axis=2)

        image = torch.tensor(t_img, dtype=torch.float)
        heatmap = torch.tensor(np.transpose(heatmap, (2, 0, 1)), dtype=torch.float)
        pts = torch.tensor(pts, dtype=torch.float)
        meta = {'index': idx, 'pts': pts}

        return image, heatmap, meta


class IDRIDOD(data.Dataset):
    """
    REFUGE2018 data pipline (single point version)
    """

    def __init__(self, cfg, mode='train'):
        mode = mode.lower()
        assert mode == 'train' or mode == 'test'

        self.sigma = cfg.MODEL.SIGMA
        self.label_type = cfg.MODEL.TARGET_TYPE
        self.stride = cfg.MODEL.STRIDE

        self.mode = mode
        self.output_size = cfg.MODEL.HEATMAP_SIZE
        self.path = cfg.DATASET.ROOT  # base path

        if self.mode == 'train':
            self.od = pd.read_csv(os.path.join(self.path, self.mode, 'groundtruths', cfg.DATASET.TRAINSET_OD))
        elif self.mode == 'test':
            self.od = pd.read_csv(os.path.join(self.path, self.mode, 'groundtruths', cfg.DATASET.TESTSET_OD))

        fovea_num = len(self.od['Image No'].dropna(axis=0))

        self.length = fovea_num
        self.parts = cfg.MODEL.NUM_JOINTS

    def __len__(self):
        return self.length

    def __data_pipline(self, img, ldmarks):
        transform = None
        if self.mode == 'train':
            transform = A.Compose(
                [
                    A.Resize(height=self.output_size[0], width=self.output_size[1], p=1),
                    A.Crop(x_min=40, y_min=0, x_max=self.output_size[1] - 76, y_max=self.output_size[0], p=1),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.ToFloat(p=1),
                ],
                keypoint_params=A.KeypointParams(format='xy')
            )
        elif self.mode == 'test':
            transform = A.Compose(
                [
                    A.Resize(height=self.output_size[0], width=self.output_size[1], p=1),
                    A.Crop(x_min=40, y_min=0, x_max=self.output_size[1] - 76, y_max=self.output_size[0], p=1),
                    A.ToFloat(p=1),
                ],
                keypoint_params=A.KeypointParams(format='xy')
            )
        transformed = transform(image=img, keypoints=ldmarks)

        return transformed

    def __getitem__(self, idx):
        name = self.od.iloc[idx, 0]
        img = Image.open(os.path.join(self.path, self.mode, "images", name + ".jpg")).convert("RGB")

        od_x, od_y = self.od.iloc[idx, 1], self.od.iloc[idx, 2]

        keypoints = [
            (od_x, od_y)
        ]

        transformed = self.__data_pipline(np.array(img), keypoints)
        t_img = transformed['image'].astype(np.float32)
        H, W, _ = t_img.shape

        t_keypoints = transformed['keypoints']
        t_img = t_img.transpose([2, 0, 1])

        # Generate heatmaps
        heatmap = np.zeros((H // self.stride, W // self.stride, self.parts + 1), np.float32)

        # Anatomical landmarks
        pts = np.array(t_keypoints, dtype=np.float32).reshape(-1, 2)
        tpts = np.array(t_keypoints, dtype=np.float32).reshape(-1, 2) / self.stride

        for i in range(self.parts):
            kernel = gaussian_kernel(size_h=H // self.stride,
                                     size_w=W // self.stride,
                                     center_x=tpts[i][0], center_y=tpts[i][1],
                                     sigma=self.sigma)
            kernel[kernel > 1] = 1
            kernel[kernel < 0.01] = 0
            heatmap[:, :, i + 1] = kernel

        # Generate the heatmap of background
        heatmap[:, :, 0] = 1.0 - np.max(heatmap[:, :, 1:], axis=2)

        image = torch.tensor(t_img, dtype=torch.float)
        heatmap = torch.tensor(np.transpose(heatmap, (2, 0, 1)), dtype=torch.float)
        pts = torch.tensor(pts, dtype=torch.float)
        meta = {'index': idx, 'pts': pts}

        return image, heatmap, meta
