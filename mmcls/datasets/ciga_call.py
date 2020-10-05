# -*- coding: utf-8 -*-
# @Time    : 2020/9/29 下午8:33
# @Author  : zxq
# @File    : ciga_call.py
# @Software: PyCharm

import mmcv
import numpy as np

from .builder import DATASETS
from .base_dataset import BaseDataset


@DATASETS.register_module()
class CigaretteCallDataset(BaseDataset):

    # BaseDataset的__init__函数调用了此函数
    def load_annotations(self):
        assert isinstance(self.ann_file, str)
        data_infos = []
        with open(self.ann_file) as f:
            samples = [x.strip().split(' ') for x in f.readlines()]
            for filename, gt_label in samples:
                info = {'img_prefix': self.data_prefix}
                info['img_info'] = {'filename': filename}
                info['gt_label'] = np.array(gt_label, dtype=np.int64)
                data_infos.append(info)
            return data_infos