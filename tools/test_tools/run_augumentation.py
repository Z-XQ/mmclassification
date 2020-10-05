# -*- coding: utf-8 -*-
# @Time    : 2020/10/4 下午7:23
# @Author  : zxq
# @File    : run_augumentation.py
# @Software: PyCharm
import numpy as np
import cv2
from matplotlib import pyplot as plt
from IPython.display import display, HTML

from albumentations import (VerticalFlip, HorizontalFlip, Flip, RandomRotate90, Rotate, ShiftScaleRotate, CenterCrop,
                            OpticalDistortion, GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
                            RGBShift, RandomBrightness, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise, CLAHE,
                            ChannelShuffle, InvertImg, RandomGamma, ToGray, PadIfNeeded, RandomBrightnessContrast,
                            ImageCompression
                            )


def show_img(img, figsize=(8, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.imshow(img)
    plt.imshow(img)
    plt.show()


def show_augmentation(img, augmenation, **params):
    params_code = ', '.join(f'{key}={value}' for key, value in params.items())
    if params_code:
        params_code += ', '
    text = HTML(
        'Use this augmentation in your code:'
        '<pre style="display:block; background-color: #eee; margin: 10px; padding: 10px;">'
        f'{augmenation.__class__.__name__}({params_code}p=0.5)'
        '</pre>'
    )
    display(text)
    show_img(img)


image = cv2.imread('/home/zxq/PycharmProjects/data/ciga_call/train/8891.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (256, 256))  # 512

show_img(image)





