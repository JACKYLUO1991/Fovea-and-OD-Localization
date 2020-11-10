#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/10/29 13:17
# @Author  : JackyLUO
# @E-mail  : lingluo@stumail.neu.edu.cn
# @Site    : 
# @File    : overall_test.py
# @Software: PyCharm

import torch

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from models.hrnet import get_fovea_net

from datasets import get_dataset
from configuration import config, update_config

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--cfg', help='experiment configuration filename',
                    default='./configuration/fovea_idrid_hrnet_w18.yaml', type=str)
args = parser.parse_args()
update_config(config, args)

dataset_type = get_dataset(config)
dataset = dataset_type(config, mode='test')
# print(len(dataset))
#
dataset[0]

# model = get_fovea_net(config)
# data = torch.rand(1, 3, 356, 536)
# out = model(data)
# print(out.size())
