#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/10/26 14:20
# @Author  : JackyLUO
# @E-mail  : lingluo@stumail.neu.edu.cn
# @Site    : 
# @File    : main.py
# @Software: PyCharm

from argparse import ArgumentParser

from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint

from configuration import config, update_config
from datasets import get_dataset
from models import HeatmapModel


def cli_main():
    pl.seed_everything(2020)

    parser = ArgumentParser()
    parser.add_argument('--cfg', help='experiment configuration filename',
                        required=True, type=str)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    update_config(config, args)

    # # Enable cudnn
    # cudnn.benchmark = config.CUDNN.BENCHMARK
    # cudnn.determinstic = config.CUDNN.DETERMINISTIC
    # cudnn.enabled = config.CUDNN.ENABLED

    # Dataset setup
    dataset_type = get_dataset(config)
    dataset = dataset_type(config, mode='train')
    train_dataset, val_dataset = random_split(dataset, [383, 30])  # custom
    # test_dataset = dataset_type(config, mode='test')

    train_loader = DataLoader(train_dataset, batch_size=config.TRAIN.BATCH_SIZE_PER_GPU, shuffle=config.TRAIN.SHUFFLE,
                              num_workers=config.WORKERS, pin_memory=config.PIN_MEMORY)
    val_loader = DataLoader(val_dataset, batch_size=config.TRAIN.BATCH_SIZE_PER_GPU, num_workers=config.WORKERS,
                            pin_memory=config.PIN_MEMORY)
    # test_loader = DataLoader(test_dataset, batch_size=config.TRAIN.BATCH_SIZE_PER_GPU, num_workers=config.WORKERS,
    #                          pin_memory=config.PIN_MEMORY, drop_last=True)

    # Define model
    model = HeatmapModel(config)
    # trainer = pl.Trainer.from_argparse_args(args)
    # accumulate_grad_batches: Accumulated gradients runs K small batches of size N before doing
    #                   a backwards pass. The effect is a large effective batch size of size KxN.
    trainer = pl.Trainer(gpus=list(config.GPUS), max_epochs=config.TRAIN.END_EPOCH,
                         profiler=True, deterministic=True)
    # trainer = pl.Trainer()
    trainer.fit(model, train_loader, val_loader)
    # trainer.test(test_dataloaders=test_loader)


if __name__ == '__main__':
    cli_main()
