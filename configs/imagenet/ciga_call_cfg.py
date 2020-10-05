# -*- coding: utf-8 -*-
# @Time    : 2020/9/29 下午10:18
# @Author  : zxq
# @File    : ciga_call_cfg.py
# @Software: PyCharm

# ----------------------model------------------------------
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(3,),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=3,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    )
)

# ------------------------------------------Dataset-----------------------------------
dataset_type = 'CigaretteCallDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

_Albu_transforms = [
    dict(
        type='RandomBrightnessContrast',
        p=0.2),
    dict(
        type='HueSaturationValue',
        p=0.2),
    dict(
        type='ImageCompression',
        quality_lower=20,
        quality_upper=100,   # quality: [lower, upper]
        p=0.2),
    dict(
        type='HorizontalFlip',
        p=0.5),
    dict(
        type='OpticalDistortion',
        p=0.2,
    ),
    dict(
        type='RandomGamma',
        p=0.2
    ),
    dict(
        type='Rotate',
        limit=[-45, 45],
        p=0.2
    ),
    dict(
        type='RGBShift',
        p=0.2
    ),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.3),
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=256),
    dict(type='Albu', transforms=_Albu_transforms),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=256),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        type='CigaretteCallDataset',
        data_prefix='/home/zxq/PycharmProjects/data/ciga_call/train',
        pipeline=train_pipeline,
        ann_file='/home/zxq/PycharmProjects/data/ciga_call/label_info.txt'),
    val=dict(
        type='CigaretteCallDataset',
        data_prefix='/home/zxq/PycharmProjects/data/ciga_call/train',
        ann_file='/home/zxq/PycharmProjects/data/ciga_call/label_info.txt',
        pipeline=test_pipeline),
    test=dict(
            type='CigaretteCallDataset',
            data_prefix='/home/zxq/PycharmProjects/data/ciga_call/test',
            ann_file='/home/zxq/PycharmProjects/data/ciga_call/label_info.txt',
            pipeline=test_pipeline)
    )
evaluation = dict(interval=1, metric='accuracy')
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=[30, 60, 90], warmup=None)
total_epochs = 100
checkpoint_config = dict(interval=1)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = '../work_dir/version02'
seed = 0
gpu_ids = range(0, 1)
