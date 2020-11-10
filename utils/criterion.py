#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/11/5 11:18
# @Author  : JackyLUO
# @E-mail  : lingluo@stumail.neu.edu.cn
# @Site    : 
# @File    : criterion.py
# @Software: PyCharm

import torch
import torch.nn as nn


class AdaptiveWingLoss(nn.Module):
    """
    Based on paper: Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression
    """

    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1, reduction='mean'):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred, target):
        delta_y = (target - pred).abs()
        delta_y1 = delta_y[delta_y < self.theta]
        delta_y2 = delta_y[delta_y >= self.theta]
        y1 = target[delta_y < self.theta]
        y2 = target[delta_y >= self.theta]
        loss1 = self.omega * torch.log(1 + torch.pow(delta_y1 / self.epsilon, self.alpha - y1))
        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))) * (self.alpha - y2) * (
            torch.pow(self.theta / self.epsilon, self.alpha - y2 - 1)) * (1 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))
        loss2 = A * delta_y2 - C

        if self.reduction == 'mean':
            return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))

        return loss1.sum() + loss2.sum()
