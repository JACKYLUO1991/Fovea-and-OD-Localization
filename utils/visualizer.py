#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/10/26 16:46
# @Author  : JackyLUO
# @E-mail  : lingluo@stumail.neu.edu.cn
# @Site    : 
# @File    : visualizer.py
# @Software: PyCharm

import cv2 as cv
from matplotlib import pyplot as plt


def vis_keypoints(image, keypoints, color=(0, 0, 255), diameter=2):
    image = image.copy()

    for (x, y) in keypoints:
        cv.circle(image, (int(x), int(y)), diameter, color, -1)

    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.imshow(image)
    plt.show()


def save_vis_keypoints(img, keypoints, color=(255, 255, 255), diameter=6, save_path=None):
    image = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    for i, k in enumerate(keypoints):
        # if i == 0:
        #     cv.circle(image, (int(k[0]), int(k[1])), diameter, color, -1)
        # else:
        #     cv.circle(image, (int(k[0]), int(k[1])), diameter, (255, 255, 255), -1)

        # Drawing a cross
        if i == 0:
            cv.line(image, (int(k[0]) - diameter, int(k[1])), (int(k[0]) + diameter, int(k[1])), color, 2)
            cv.line(image, (int(k[0]), int(k[1]) - diameter), (int(k[0]), int(k[1]) + diameter), color, 2)
        else:
            color = (0, 0, 0)
            cv.line(image, (int(k[0]) - diameter, int(k[1])), (int(k[0]) + diameter, int(k[1])), color, 2)
            cv.line(image, (int(k[0]), int(k[1]) - diameter), (int(k[0]), int(k[1]) + diameter), color, 2)
    cv.imwrite(save_path, image)


def display_keypoints(img, keypoints, color=(255, 255, 255), diameter=7):
    image = cv.cvtColor(img, cv.COLOR_RGB2BGR)

    if len(keypoints) == 2:
        for j in keypoints:
            for i, k in enumerate(j):
                if i == 0:
                    cv.line(image, (int(k[0]) - diameter, int(k[1])), (int(k[0]) + diameter, int(k[1])), color, 2)
                    cv.line(image, (int(k[0]), int(k[1]) - diameter), (int(k[0]), int(k[1]) + diameter), color, 2)
                else:
                    cv.line(image, (int(k[0]) - diameter, int(k[1])), (int(k[0]) + diameter, int(k[1])), (0, 0, 0), 2)
                    cv.line(image, (int(k[0]), int(k[1]) - diameter), (int(k[0]), int(k[1]) + diameter), (0, 0, 0), 2)
    else:
        for i, k in enumerate(keypoints[0]):
            if i == 0:
                cv.line(image, (int(k[0]) - diameter, int(k[1])), (int(k[0]) + diameter, int(k[1])), color, 2)
                cv.line(image, (int(k[0]), int(k[1]) - diameter), (int(k[0]), int(k[1]) + diameter), color, 2)
            else:
                cv.line(image, (int(k[0]) - diameter, int(k[1])), (int(k[0]) + diameter, int(k[1])), (0, 0, 0), 2)
                cv.line(image, (int(k[0]), int(k[1]) - diameter), (int(k[0]), int(k[1]) + diameter), (0, 0, 0), 2)

    return image
