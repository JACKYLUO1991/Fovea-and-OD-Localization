#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/10/27 20:33
# @Author  : JackyLUO
# @E-mail  : lingluo@stumail.neu.edu.cn
# @Site    : 
# @File    : evaluation.py
# @Software: PyCharm

import math
import numpy as np
import numpy.ma as ma
from scipy.ndimage.measurements import center_of_mass

import torch


# def get_preds(scores):
#     """
#     get predictions from score maps in torch Tensor
#     return type: torch.LongTensor
#     """
#     assert scores.dim() == 4, 'Score maps should be 4-dim'
#     maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)
#
#     maxval = maxval.view(scores.size(0), scores.size(1), 1)
#     idx = idx.view(scores.size(0), scores.size(1), 1) + 1
#
#     preds = idx.repeat(1, 1, 2).float()
#
#     preds[:, :, 0] = (preds[:, :, 0] - 1) % scores.size(3) + 1
#     preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / scores.size(3)) + 1
#
#     pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
#     preds *= pred_mask
#
#     return preds
#
#
# def decode_preds(output, res):
#     coords = get_preds(output)
#
#     coords = coords.cpu()
#     # pose-processing
#     for n in range(coords.size(0)):
#         for p in range(coords.size(1)):
#             hm = output[n][p]
#             px = int(math.floor(coords[n][p][0]))
#             py = int(math.floor(coords[n][p][1]))
#             if (px > 1) and (px < res[0]) and (py > 1) and (py < res[1]):
#                 diff = torch.Tensor([hm[py - 1][px] - hm[py - 1][px - 2], hm[py][px - 1] - hm[py - 2][px - 1]])
#                 coords[n][p] += diff.sign() * .25
#     coords += 0.5
#     preds = coords.clone()
#
#     if preds.dim() < 3:
#         preds = preds.view(1, preds.size())
#
#     return preds


def get_key_points(heatmap, o_size, per=None):
    """
    Get all key points from heatmap6.
    :param per: Take N (default: 1) percent of the pixels
    :param o_size: Output image size
    :param heatmap: The heatmap6 of CPM cpm.
    :return: All key points of the original image.
    """
    # Get final heatmap
    heatmap = np.asarray(heatmap.cpu().data)[0]
    H, W, _ = heatmap.shape

    key_points = []
    # Get k key points from heatmap
    for i in heatmap[1:]:
        # Get the coordinate of key point in the heatmap
        y, x = np.unravel_index(np.argmax(i), i.shape)
        # Get the centroid
        if per is not None:
            i_sort = np.sort(i.flatten())[::-1]
            indice = int(H * W * per)
            threshold = i_sort[indice - 1]
            mask = i < threshold
            mx = ma.masked_array(i, mask=mask).filled(0)
            y, x = center_of_mass(mx)

        # Calculate the scale to fit original image
        scale_x = o_size[0] / i.shape[0]
        scale_y = o_size[1] / i.shape[1]
        x = int(x * scale_x)
        y = int(y * scale_y)

        key_points.append([x, y])

    return np.asarray(key_points)


def distance_error(pred_coords, orig_coords, stride=8):
    xp, yp = pred_coords
    xo, yo = orig_coords

    dist = np.sqrt((xo - xp) ** 2 + (yo - yp) ** 2)

    return dist * stride


def test_time_augmentation(inputs, model):
    inputs_flip = torch.flip(inputs, dims=[-1])
    return model(inputs_flip)
