#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/10/27 21:11
# @Author  : JackyLUO
# @E-mail  : lingluo@stumail.neu.edu.cn
# @Site    :
# @File    : test.py
# @Software: PyCharm

import torch
# import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

from argparse import ArgumentParser

from datasets import get_dataset
from models import HeatmapModel
from configuration import config, update_config

# from utils import decode_preds
from utils import get_key_points
from utils import distance_error
from utils import save_vis_keypoints, display_keypoints
from utils import test_time_augmentation
# from utils import tensor2im

import os
import numpy as np
from tqdm import tqdm
import pandas as pd

import cv2 as cv


def parse_args():
    parser = ArgumentParser(description='Train Face Alignment')

    parser.add_argument('--cfg', help='experiment configuration filename',
                        default='./configuration/fovea_idrid_hrnet_w18.yaml', type=str)
    parser.add_argument('--model-file', help='model parameters',
                        default='./lightning_logs/version_0/checkpoints/epoch=28.ckpt',
                        type=str)
    parser.add_argument('--out', help='image out path', default='results', type=str)
    parser.add_argument('--test-aug', help='whether use test augmentation or not', action='store_true')
    args = parser.parse_args()
    update_config(config, args)

    return args


def main():
    args = parse_args()

    # cudnn.benchmark = config.CUDNN.BENCHMARK
    # cudnn.determinstic = config.CUDNN.DETERMINISTIC
    # cudnn.enabled = config.CUDNN.ENABLED
    #
    # config.defrost()
    # config.MODEL.INIT_WEIGHTS = False
    # config.freeze()

    model = HeatmapModel.load_from_checkpoint(args.model_file)
    model.cuda()
    model.eval()

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    # model = get_fovea_net(config).cuda()  # one GPU!
    # checkpoint = torch.load(args.model_file)
    # model.load_state_dict(checkpoint['state_dict'])

    # if 'state_dict' in state_dict.keys():
    #     state_dict = state_dict['state_dict']
    #     model.load_state_dict(state_dict)
    # else:
    #     raise FileNotFoundError("No files were found")

    dataset_type = get_dataset(config)
    test_dataset = dataset_type(config, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=config.TEST.BATCH_SIZE_PER_GPU, num_workers=config.WORKERS,
                             pin_memory=config.PIN_MEMORY)
    H = config.MODEL.HEATMAP_SIZE[0]
    W = config.MODEL.HEATMAP_SIZE[1] - 116  # remove useless interference

    test_augmentation = args.test_aug

    od_euclidean_error = []
    fovea_euclidean_error = []
    compare_preds = None

    with torch.no_grad():
        for i, (inputs, heatmaps, meta) in tqdm(enumerate(test_loader)):
            # img = tensor2im(inputs.squeeze(0))
            img = transforms.ToPILImage()(inputs.squeeze(0))
            img = np.asarray(img)

            inputs = inputs.cuda()
            score_map = model(inputs)
            gt = meta['pts'].numpy()
            # preds = decode_preds(score_map, config.MODEL.HEATMAP_SIZE)

            # Test time augmentation (TTA)
            if test_augmentation:
                score_map_flip = test_time_augmentation(inputs, model)

            # Post-processing
            preds = get_key_points(score_map, (H, W), per=0.01)

            if test_augmentation:
                preds_flip = get_key_points(score_map_flip, (H, W), per=0.01)
                preds_flip[:, 0] = W - preds_flip[:, 0]
                preds = (preds + preds_flip) / 2

            # Multi-task or not
            if heatmaps.size()[1] == 3:
                gt_od = gt[0][0]
                preds_od = preds[0].tolist()
                gt_fovea = gt[0][1]
                preds_fovea = preds[1].tolist()
                compare_preds = [
                    [gt_fovea, preds_fovea], [gt_od, preds_od]
                ]

                error_od = distance_error(gt_od, preds_od, stride=8)
                od_euclidean_error.append(error_od)
            else:
                gt_fovea = gt[0][0]
                preds_fovea = preds[0].tolist()
                compare_preds = [
                    [gt_fovea,
                     preds_fovea]
                ]

            error_fovea = distance_error(gt_fovea, preds_fovea, stride=8)
            fovea_euclidean_error.append(error_fovea)
            image = display_keypoints(img, compare_preds)
            cv.imwrite(args.out + '/' + f'test{i}.png', image)

    if len(compare_preds) == 2:
        od_mean_euclidean_error = np.array(od_euclidean_error, dtype=np.float32)
        df = pd.DataFrame(od_mean_euclidean_error, columns=['error'])
        df.to_csv("./od.csv")
        print(f"OD Euclidean Distance: {od_mean_euclidean_error.mean()}")

    fovea_mean_euclidean_error = np.array(fovea_euclidean_error, dtype=np.float32)
    df = pd.DataFrame(fovea_mean_euclidean_error, columns=['error'])
    df.to_csv("./fovea.csv")

    print(f"Fovea Euclidean Distance: {fovea_mean_euclidean_error.mean()}")


if __name__ == '__main__':
    main()
