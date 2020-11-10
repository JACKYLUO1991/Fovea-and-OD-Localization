#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/10/26 19:45
# @Author  : JackyLUO
# @E-mail  : lingluo@stumail.neu.edu.cn
# @Site    : 
# @File    : utils.py
# @Software: PyCharm

import numpy as np

import torch
import torch.optim as optim


# def generate_target(img, pt, sigma, label_type='Gaussian'):
#     # Check that any part of the gaussian is in-bounds
#     tmp_size = sigma * 3
#     ul = [int(pt[0] - tmp_size), int(pt[1] - tmp_size)]
#     br = [int(pt[0] + tmp_size + 1), int(pt[1] + tmp_size + 1)]
#     if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
#             br[0] < 0 or br[1] < 0):
#         return img
#
#     # Generate gaussian
#     size = 2 * tmp_size + 1
#     x = np.arange(0, size, 1, np.float32)
#     y = x[:, np.newaxis]
#     x0 = y0 = size // 2
#     # The gaussian is not normalized, we want the center value to equal 1
#     if label_type == 'Gaussian':
#         g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
#     else:
#         g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)
#
#     # Usable gaussian range
#     g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
#     g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
#     # Image range
#     img_x = max(0, ul[0]), min(br[0], img.shape[1])
#     img_y = max(0, ul[1]), min(br[1], img.shape[0])
#
#     img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
#
#     return img


def gaussian_kernel(size_w, size_h, center_x, center_y, sigma):
    grid_y, grid_x = np.mgrid[0: size_h, 0: size_w]
    D2 = (grid_x - center_x) ** 2 + (grid_y - center_y) ** 2

    return np.exp(-D2 / 2.0 / sigma / sigma)


def get_optimizer(cfg, model):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR
        )
    elif cfg.TRAIN.OPTIMIZER == 'rmsprop':
        optimizer = optim.RMSprop(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            # alpha=cfg.TRAIN.RMSPROP_ALPHA,
            # centered=cfg.TRAIN.RMSPROP_CENTERED
        )

    return optimizer


def tensor2im(image_tensor, imtype=np.uint8):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    for i in range(len(mean)):
        image_numpy[i] = image_numpy[i] * std[i] + mean[i]
    image_numpy = np.clip(image_numpy * 255, 0, 255)
    image_numpy = np.transpose(image_numpy, (1, 2, 0))

    return image_numpy.astype(imtype)
