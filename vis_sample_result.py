# -*- coding: utf-8 -*-
# @Time    : 2019/3/18 15:48
# @Author  : Jason
# @Email   : 1358681631@qq.com
# @File    : vis_sample_result.py.py
# @Software: PyCharm
import cv2
import os

img = cv2.imread(os.path.expanduser("/sample_result/videos14/global_parsing/000000000001.png"))
print(img.shape)
