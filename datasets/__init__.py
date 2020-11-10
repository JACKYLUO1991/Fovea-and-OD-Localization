#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/10/27 12:24
# @Author  : JackyLUO
# @E-mail  : lingluo@stumail.neu.edu.cn
# @Site    :
# @File    : __init__.py
# @Software: PyCharm


from .idrid import IDRID, IDRIDFOVEA, IDRIDOD

__all__ = ['IDRID', 'get_dataset', 'IDRIDFOVEA', 'IDRIDOD']


def get_dataset(config):
    if config.DATASET.DATASET == 'IDRID':
        return IDRID
    elif config.DATASET.DATASET == 'IDRIDOD':
        return IDRIDOD
    elif config.DATASET.DATASET == 'IDRIDFOVEA':
        return IDRIDFOVEA
    else:
        raise NotImplemented()
