# -*- coding: utf-8 -*-
# @Time    : 2020/9/29 下午9:07
# @Author  : zxq
# @File    : train_tmp.py
# @Software: PyCharm
import os.path as osp
import mmcv
from mmcv import Config

from mmcls.apis import train_model, set_random_seed
from mmcls.datasets import build_dataset
from mmcls.models import build_classifier


# class CustomerTrain(object):
#     def __init__(self, cfg):

        # # Modify dataset type and path
        # cfg.dataset_type = 'CigaretteCallDataset'
        # cfg.data.train.type = 'CigaretteCallDataset'
        # cfg.data.train.data_prefix = '/home/zxq/PycharmProjects/data/ciga_call/train'
        # cfg.data.train.ann_file = '/home/zxq/PycharmProjects/data/ciga_call/label_info.txt'
        #
        # cfg.data.val.type = 'CigaretteCallDataset'
        # cfg.data.val.data_prefix = '/home/zxq/PycharmProjects/data/ciga_call/train'
        # cfg.data.val.ann_file = '/home/zxq/PycharmProjects/data/ciga_call/label_info.txt'
        #
        # cfg.data.test.type = 'CigaretteCallDataset'
        # cfg.data.test.data_prefix = '/home/zxq/PycharmProjects/data/ciga_call/train'
        # cfg.data.test.ann_file = '/home/zxq/PycharmProjects/data/ciga_call/label_info.txt'
        #
        # # modify num classes of the model
        # cfg.model.head.num_classes = 3  # 检测的类别数
        # # We can still use the pre-trained Mask RCNN model though we do not need to
        # # use the mask branch
        # # cfg.load_from = '../weights/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
        #
        # # Set up working dir to save files and logs.
        # cfg.work_dir = '../work_dir'
        #
        # # The original learning rate (LR) is set for 8-GPU training.
        # # We divide it by 8 since we only use one GPU.
        # # cfg.optimizer.lr = 0.02 / 8
        # cfg.lr_config.warmup = None
        # # cfg.log_config.interval = 10
        #
        # # Change the evaluation metric since we use customized dataset.
        # # cfg.evaluation.metric = 'mAP'
        # # We can set the evaluation interval to reduce the evaluation times
        # # cfg.evaluation.interval = 12
        # # We can set the checkpoint saving interval to reduce the storage cost
        # # cfg.checkpoint_config.interval = 12
        #
        # # Set seed thus the results are more reproducible
        # cfg.seed = 0
        # set_random_seed(0, deterministic=False)
        # cfg.gpu_ids = range(1)
        # self.cfg = cfg
        #
        # # We can initialize the logger for training and have a look
        # # at the final config used for training
        # print(f'Config:\n{cfg.pretty_text}')


if __name__ == '__main__':
    cfg = Config.fromfile('../configs/imagenet/ciga_call_cfg.py')
    # customer_train = CustomerTrain(cfg)
    # cfg = customer_train.cfg
    # Build dataset
    datasets = [build_dataset(cfg.data.train)]

    # Build the detector
    model = build_classifier(cfg.model)
    # Add an attribute for visualization convenience
    # model.CLASSES = datasets[0].CLASSES

    # Create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    train_model(model, datasets, cfg, distributed=False, validate=True)