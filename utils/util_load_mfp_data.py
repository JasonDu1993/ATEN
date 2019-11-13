# -*- coding: utf-8 -*-
# @Time    : 2019/11/13 10:44
# @Author  : Jason
# @Email   : 1358681631@qq.com
# @File    : util_load_mfp_data.py
# @Software: PyCharm
import cv2
import numpy as np
import os
from utils.util import resize_image, resize_mask, resize_part_mfp


def get_scale(image_shape, config):
    h, w, c = image_shape
    scale = 1
    # Scale?
    if config.IMAGE_MIN_DIM:
        # Scale up but not down
        scale = max(1, config.IMAGE_MIN_DIM / min(h, w))
    # Does it exceed max dim?
    if config.IMAGE_MAX_DIM:
        image_max = max(h, w)
        if round(image_max * scale) > config.IMAGE_MAX_DIM:
            scale = config.IMAGE_MAX_DIM / image_max
    return scale


def load_pre_image_names(image_name, key_num=3, gap=1):
    """

    Args:
        image_name: image_name includes video_name/image_id, for example videos45/000000000176
        key_num: int, default is 3
        gap: int, default is 1

    Returns:
        pre_image_names: list, the value is list, len=2, which represent video_name, pre_image_id

    """
    assert key_num * (gap + 1) < 10, "key_num(%d) * (gap(%d) + 1) >= 10" % (key_num, gap)
    video_name, image_id = image_name.strip().split("/")
    pre_image_names = []
    if image_id.endswith("000000000001"):
        image_id_int = int(image_id) + 1
        for i in range(key_num):
            index_gap = i * (gap + 1)
            pre_image_id = "%012d" % (image_id_int + index_gap)
            pre_image_names.append([video_name, pre_image_id])
        return pre_image_names[::-1]
    image_id_int = int(image_id) - 1
    for i in range(key_num):
        index_gap = i * (gap + 1)
        pre_image_id = "%012d" % (image_id_int - index_gap)
        pre_image_names.append([video_name, pre_image_id])
    return pre_image_names


def load_pre_image_datas(image_name, pre_image_names, config, pre_image_dir, pre_predict_data_dir):
    """

    Args:
        image_name: image_name includes video_name/image_id, for example videos45/000000000176
        pre_image_names: list, the value is list, len=2, which represent video_name, pre_image_id
        config: The model config object
        pre_image_dir: str, the pre image dir
        pre_predict_data_dir: str, the pre image predicted dir

    Returns:
        pre_images: list, the value is numpy.ndarray, shape [resize_height=512, resize_width=512, 3(BGR)]
        pre_masks: list, the value is numpy.ndarray, shape [resize_height, resize_width, 1],
            which value include 0 ~ num_person, 0 is bg, 1 ~ num_person is the person label
        pre_parts: list, the value is numpy.ndarray, shape [resize_height, resize_width, num_class=20],
            which value include 0 ~ 19, 0 is bg, 1 ~ 19 is the person part label

    """
    video_name, image_id = image_name.strip().split("/")
    pre_images = []
    pre_masks = []
    pre_parts = []
    scale = 1
    for pre_video_name, pre_image_id in pre_image_names:
        pre_image_path = os.path.join(pre_image_dir, "adjacent_frames", pre_video_name, image_id, pre_image_id + ".jpg")
        pre_mask_path = os.path.join(pre_predict_data_dir, "vp_results", pre_video_name, "instance_segmentation",
                                     pre_image_id + ".png")
        pre_part_path = os.path.join(pre_predict_data_dir, "vp_results", pre_video_name, "global_parsing",
                                     pre_image_id + ".png")
        pre_image = cv2.imread(pre_image_path)  # shape [h=720, w=1080, 3(bgr)]
        pre_mask = cv2.imread(pre_mask_path, flags=cv2.IMREAD_GRAYSCALE)  # shape [h=720, w=1080]
        pre_part_tmp = cv2.imread(pre_part_path, flags=cv2.IMREAD_GRAYSCALE)  # shape [h=720, w=1080]
        pre_part = np.zeros(pre_part_tmp.shape + (config.NUM_PART_CLASS,))
        import time
        t0 = time.time()
        for i in range(1, config.NUM_PART_CLASS):
            pre_part[pre_part_tmp == i] = 1
        # print("pre_part generate cost:", time.time() - t0, "s")
        pre_image, window, scale, padding = resize_image(pre_image, max_dim=config.PRE_IMAGE_SHAPE[0],
                                                         padding=config.IMAGE_PADDING, isopencv=True)
        pre_mask = resize_mask(pre_mask, scale, padding, isopencv=True)[:, :, np.newaxis]  # shape [512, 512,1]
        pre_part = resize_part_mfp(pre_part, scale, padding, isopencv=True)  # [512,512,20]
        pre_images.append(pre_image[np.newaxis, ...])
        pre_masks.append(pre_mask[np.newaxis, ...])
        pre_parts.append(pre_part[np.newaxis, ...])
    return pre_images, pre_masks, pre_parts


def load_pre_image_boxes(pre_image_names, scale, pre_predict_data_dir):
    """

    Args:
        pre_image_names: list, the value is list, len=2, which represent video_name, pre_image_id
        scale: The scale factor used to resize the image
        pre_predict_data_dir: str, the pre image predicted dir

    Returns:
        pre_boxes: list, the value is list, len is 4, which represent y1, x1, y2, x2.

    """
    pre_boxes = []
    for pre_video_name, pre_image_id in pre_image_names:
        boxes_path = os.path.join(pre_predict_data_dir, "vp_results", pre_video_name, "instance_segmentation",
                                  pre_image_id + ".txt")
        with open(boxes_path, "r") as f:
            for line in f.readlines():
                y1, x1, y2, x2 = list(map(int, line.strip().split(" ")[1:]))
                pre_boxes.append([round(y1 * scale), round(x1 * scale), round(y2 * scale), round(x2 * scale)])
    return pre_boxes
