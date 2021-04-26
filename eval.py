# -*- coding: utf-8 -*-
# @Time    : 2018/6/11 15:54
# @Author  : zhoujun
import os
import cv2
import torch
import shutil
import numpy as np
import time
from tqdm.auto import tqdm
from utils import cal_recall_precison_f1, draw_bbox, load_json
from demo import demo_Model

torch.backends.cudnn.benchmark = True


def main(model_path, img_folder, save_path, gpu_id):
    if os.path.exists(save_path):
        shutil.rmtree(save_path, ignore_errors=True)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_img_folder = os.path.join(save_path, 'img')
    if not os.path.exists(save_img_folder):
        os.makedirs(save_img_folder)
    save_txt_folder = os.path.join(save_path, 'result')
    if not os.path.exists(save_txt_folder):
        os.makedirs(save_txt_folder)
    img_paths = [os.path.join(img_folder, x) for x in os.listdir(img_folder)]

    config = load_json('./config/config_pan.json')
    model = demo_Model(config)
    model.load_model(model_path)

    total_frame = 0.0
    total_time = 0.0
    for i, img_path in enumerate(img_paths):
        print(i)
        if i in [243]:
            continue
        img_name = os.path.basename(img_path).split('.')[0]
        save_name = os.path.join(save_txt_folder, 'res_' + img_name + '.txt')
        _, boxes_list, t = model.demo(img_path, 320)
        total_frame += 1
        total_time += t
        img = draw_bbox(img_path, boxes_list, color=(0, 0, 255))
        cv2.imwrite(os.path.join(save_img_folder, '{}.jpg'.format(img_name)), img)
        np.savetxt(save_name, boxes_list.reshape(-1, 8), delimiter=',', fmt='%d')
    print('fps:{}'.format(total_frame / total_time))
    return save_txt_folder


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = str('0')
    model_path = './output/epoch_5234_iter_178000.pth'
    img_path = '/root/Storage/datasets/ICDAR2015/test/images'
    gt_path = '/root/Storage/datasets/ICDAR2015/test/gt'
    save_path = './output/result/'
    gpu_id = 0

    main(model_path, img_path, save_path, gpu_id=gpu_id)
    result = cal_recall_precison_f1(gt_path=gt_path, result_path='./output/result/result')
    print(result)
