# -*- coding: utf-8 -*-
# @Time    : 2019/3/22 11:51
# @Author  : Jason
# @Email   : 1358681631@qq.com
# @File    : submit_testfiles.py.py
# @Software: PyCharm
# 该文件用于之前修改预测的文件存储顺序不是提交所需顺序
import os
import shutil

dir_path = "/home/sk49/workspace/zhoudu/ATEN/vis/test_vip_singleframe_20181229ma_epoch086"
submit_path = "/home/sk49/workspace/zhoudu/ATEN/vis/test_vip_singleframe_20181229ma_epoch086_submit"

videos = os.listdir(dir_path)
print(videos)
for video in videos:
    color, gray, instance_part = os.listdir(os.path.join(dir_path, video))
    global_parsing = os.path.join(submit_path, video, "global_parsing")
    instance_parsing = os.path.join(submit_path, video, "instance_parsing")
    instance_segmentation = os.path.join(submit_path, video, "instance_segmentation")
    if not os.path.exists(global_parsing):
        os.makedirs(global_parsing)
    if not os.path.exists(instance_parsing):
        os.makedirs(instance_parsing)
    if not os.path.exists(instance_segmentation):
        os.makedirs(instance_segmentation)
    color_dir = os.path.join(dir_path, video, color)
    gray_dir = os.path.join(dir_path, video, gray)
    instance_part_dir = os.path.join(dir_path, video, instance_part)

    for name in os.listdir(gray_dir):
        if name.startswith("global_"):
            new_name = name.strip().split("_")[-1]
            shutil.copy(os.path.join(gray_dir, name), os.path.join(global_parsing, new_name))
        if name.startswith("inst_"):
            new_name = name.strip().split("_")[-1]
            shutil.copy(os.path.join(gray_dir, name), os.path.join(instance_segmentation, new_name))
        if name.startswith("scores_"):
            new_name = name.strip().split("_")[-1]
            shutil.copy(os.path.join(gray_dir, name), os.path.join(instance_segmentation, new_name))
    for name in os.listdir(instance_part_dir):
        shutil.copy(os.path.join(instance_part_dir, name), os.path.join(instance_parsing, name))
