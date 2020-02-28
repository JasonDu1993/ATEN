# -*- coding: utf-8 -*-
# @Time    : 2019/5/6 17:04
# @Author  : Jason
# @Email   : 1358681631@qq.com
# @File    : extract_video_frames.py
# @Software: PyCharm
# -*- coding: utf-8 -*-

import cv2
import operator
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from time import time

dataset_dir = "/home/sk49/workspace/dataset/VIP"


def save_video_frame(mode):
    video_dir = os.path.join(dataset_dir, "videos", mode + "_videos")
    video_names = os.listdir(video_dir)
    for video_name in video_names:
        t0 = time()
        video_id = video_name.split(".")[0]
        read_video_path = os.path.join(video_dir, video_name)
        save_dir = os.path.join(dataset_dir, "videos", mode + "_videos_frames", video_id)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # save all keyframes as image
        cap = cv2.VideoCapture(str(read_video_path))
        success, frame = cap.read()
        idx = 1
        while success:
            save_path = '%s/%012d.jpg' % (save_dir, idx)
            cv2.imwrite(save_path, frame)
            idx = idx + 1
            success, frame = cap.read()
        cap.release()
        print("save:", video_name, "frame:", idx, "time:", time() - t0, "s")


def extract_video_frame(read_video_path, save_dir, name):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    t0 = time()
    # save all keyframes as image
    cap = cv2.VideoCapture(str(read_video_path))
    success, frame = cap.read()
    idx = 1
    while success:
        save_path = '%s/%s_%012d.jpg' % (save_dir, name, idx)
        cv2.imwrite(save_path, frame)
        idx = idx + 1
        success, frame = cap.read()
    cap.release()
    print("frame:", idx, "time:", time() - t0, "s")


def save_video_id(mode):
    video_dir = "/home/sk49/workspace/dataset/VIP/videos/" + mode + "_videos_frames"
    video_ids = os.listdir(video_dir)
    with open(mode + "_all_frames_id.txt", "w") as f:
        for video_id in video_ids:
            for image_id in os.listdir(os.path.join(video_dir, video_id)):
                print(image_id)
                f.write(video_id + "/" + str(image_id.strip().split(".")[0]) + "\n")


if __name__ == "__main__":
    # Video path of the source file
    t = time()
    # save_video_frame(mode="test")
    read_video_path = r"C:\test_videos\VIP_test_videos\test_videos\videos106.avi"
    save_dir = r"C:\test_videos\VIP_test_videos\test_videos\videos106"
    name = "videos106"
    extract_video_frame(read_video_path, save_dir, name)
    # save_video_id("test")
    print("total time:", time() - t, "s")
