#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/10/27 20:00
# @Author  : JackyLUO
# @E-mail  : lingluo@stumail.neu.edu.cn
# @Site    :
# @File    : model_system.py
# @Software: PyCharm

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from models import get_fovea_net
from utils import get_optimizer
from utils import AdaptiveWingLoss


class HeatmapModel(pl.LightningModule):
    """
    Define processing pipline
    """

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.model = get_fovea_net(cfg)
        self.cfg = cfg

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y, reduction='sum')
        # loss = AdaptiveWingLoss(reduction='sum')(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y, reduction='sum')
        # loss = AdaptiveWingLoss(reduction='sum')(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

    # def validation_epoch_end(self, validation_step_outputs):
    #     avgLoss = torch.stack([x['val_loss'] for x in validation_step_outputs]).mean()
    #     avgDis = torch.stack([x['val_dis'] for x in validation_step_outputs]).mean()
    #
    #     tensorboardLogs = {'val_loss': avgLoss, 'val_dis': avgDis}
    #     return {'val_loss': avgLoss, 'log': tensorboardLogs}

    # def test_step(self, batch, batch_idx):
    #     x, y, _ = batch
    #     y_hat = self.model(x)
    #     loss = F.mse_loss(y_hat, y, reduction='sum')
    #     self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = get_optimizer(self.cfg, self.model)
        return optimizer
