# -*- coding: utf-8 -*-
# @Time    : 2020/10/3 下午7:55
# @Author  : zxq
# @File    : run_test.py
# @Software: PyCharm
import os

import mmcv
from mmcv import Config

from mmcls.apis import single_gpu_test
from mmcls.datasets import build_dataset, build_dataloader
from mmcls.models import build_classifier

if __name__ == '__main__':
    cfg = Config.fromfile('../../configs/imagenet/ciga_call_cfg.py')
    # customer_train = CustomerTrain(cfg)
    # cfg = customer_train.cfg
    # Build dataset
    datasets = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        datasets,
        cfg.data.samples_per_gpu,
        cfg.data.workers_per_gpu,
        # cfg.gpus will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=False,
        round_up=True,
        seed=cfg.seed)

    # Build the detector
    model = build_classifier(cfg.model)
    # Add an attribute for visualization convenience
    # model.CLASSES = datasets[0].CLASSES

    # Create work_dir
    save_path = os.path.join(os.path.dirname(cfg.data.test.data_prefix), 'test_result')
    mmcv.mkdir_or_exist(save_path)

    results = single_gpu_test(model, data_loader, show=True, out_dir=save_path)
    print(results)

