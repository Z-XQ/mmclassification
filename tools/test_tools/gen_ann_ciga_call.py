# -*- coding: utf-8 -*-
# @Time    : 2020/9/29 下午8:41
# @Author  : zxq
# @File    : gen_ann_ciga_call.py
# @Software: PyCharm

"""
标注信息对应关系
normal 0
phone 1
smoke 2
生成标注文档：
image_name label_num
00000.jpg 1
00001.jpg 2

"""
import os

data_path = '/home/zxq/PycharmProjects/data/ciga_call/train'
save_info_path = '/home/zxq/PycharmProjects/data/ciga_call/label_info.txt'
file_pallet = open(save_info_path, mode='w')
for root, dirs, files in os.walk(data_path):
    for file_name in files:
        img_full_path = os.path.join(root, file_name)
        print(img_full_path)
        if 'normal' in img_full_path:
            file_pallet.write('{} 0\n'.format(file_name))
        elif 'phone' in img_full_path:
            file_pallet.write('{} 1\n'.format(file_name))
        elif 'smoke' in img_full_path:
            file_pallet.write('{} 2\n'.format(file_name))

file_pallet.close()