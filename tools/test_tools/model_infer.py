# -*- coding: utf-8 -*-
# @Time    : 2020/10/3 下午8:35
# @Author  : zxq
# @File    : model_infer.py
# @Software: PyCharm
import json
import os

import cv2
import mmcv
import numpy as np
import torch
from mmcv import Config

from mmcls.models import build_classifier

id_to_class = {
    0: "normal",
    1: "calling",
    2: "smoking"
}

if __name__ == '__main__':
    cfg = Config.fromfile('../../configs/imagenet/ciga_call_cfg.py')
    data_path = '/home/zxq/PycharmProjects/data/ciga_call/test'
    json_file_path = '/home/zxq/PycharmProjects/data/ciga_call/result.json'
    # json1_file_path = '/home/zxq/Downloads/result.json'
    # f = open(json_file_path, mode='r+')
    # content = json.load(f)
    json_file = open(json_file_path, mode='w')
    weight_path = '../../work_dir/version02/epoch_100.pth'
    model = build_classifier(cfg.model)
    model.eval()
    save_path = os.path.join(os.path.dirname(cfg.data.test.data_prefix), 'test_result')
    mmcv.mkdir_or_exist(save_path)

    mean_value = None
    std_value = None
    for step_ in cfg.test_pipeline:
        if step_['type'] is 'Normalize':
            mean_value = np.array(step_['mean'])
            std_value = np.array(step_['std'])
    img_name_list = os.listdir(data_path)
    save_json_content = []
    # k = 0
    for img_name in img_name_list:
        img_dir = os.path.join(data_path, img_name)
        print(img_dir)
        img = cv2.imread(img_dir)

        # 1, resize
        img_resized = mmcv.imresize(img, (256, 256))

        # 2, Normalize
        img_normalized = mmcv.imnormalize(img_resized, mean_value, std_value)

        # 3, switch dim and to tensor
        input_data = torch.Tensor(np.transpose(img_normalized, [2, 0, 1]))

        # 4, add batch dim
        batch_data = torch.unsqueeze(input_data, 0)
        # 4, infer
        model.load_state_dict(torch.load(weight_path, map_location='cpu')['state_dict'])
        model_output = model(batch_data, return_loss=False)
        output_normalized = torch.nn.functional.softmax(model_output, dim=1)
        output_numpy = output_normalized.detach().numpy()
        cls_output = np.argmax(output_numpy, axis=1)
        print(np.round(output_numpy[0][cls_output[0]], 5))

        result_json = {
            "image_name": img_name,
            "category": id_to_class[cls_output[0]],
            "score": np.round(np.float(output_numpy[0][cls_output[0]]), 5)}
        save_json_content.append(result_json)
        # json.dump(result_json, json_file, ensure_ascii=False, indent=4)
        # k += 1
        # if k > 10:
        #     break

    save_json_content.sort(key=lambda x: int(x['image_name'][:-4]))
    json.dump(save_json_content, json_file, indent=4)

