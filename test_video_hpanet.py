# -*- coding: utf-8 -*-
# @Time    : 2020/2/27 19:51
# @Author  : Jason
# @Email   : 1358681631@qq.com
# @File    : test_video_hpanet.py
# @Software: PyCharm
import os
import tensorflow as tf
import platform

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.3
session = tf.Session(config=config)
import sys
import cv2

sys.path.insert(0, os.getcwd())

from models.hpa_resfpn_c5d_edgam_e357 import HPAConfig

# from models.parsing_rcnn_model import PARSING_RCNN
from models.hpa_resfpn_c5d_edgam_e357 import HPANet
from utils import visualize
from time import time, strftime

t0 = time()
# Root directory of the project
ROOT_DIR = os.getcwd()
MACHINE_NAME = platform.node()
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "outputs")

if MACHINE_NAME == "Jason":
    # win
    MODEL_PATH = r"C:\ATEN_weights\hpa_20191128e\checkpoints\parsing_rcnn_hpa_20191128e_epoch023_loss0.881_valloss1.070.h5"
    read_video_path = r"C:\test_videos\videos1.mp4"
else:
    # linux
    MODEL_PATH = "/home/sk49/workspace/zhoudu/ATEN/outputs/hpa_20191128e/checkpoints/" \
                 "parsing_rcnn_hpa_20191128e_epoch023_loss0.881_valloss1.070.h5"
    read_video_path = r"./utils/奔跑吧兄弟赵丽颖_3s.mp4"


class InferenceConfig(HPAConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def main():
    global config
    config = InferenceConfig()
    config.display()
    vid = os.path.split(read_video_path)[-1].split(".")[0]
    print("vid:", vid)
    # Create model object in inference mode.
    model = HPANet(mode="inference", config=config, model_dir=MODEL_DIR)
    # Load weights trained on MS-COCO
    s0 = time()
    model.load_weights(MODEL_PATH, by_name=True)
    print("load model", time() - s0, "s")

    cap = cv2.VideoCapture(read_video_path)
    save_dir = os.path.join(os.path.dirname(read_video_path), vid + "_hpanet_" + strftime("%Y_%m%d_%H%M%S"))
    print("save images in", save_dir)
    success, frame = cap.read()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # else:
    #     success = False
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("fps", fps)
    # fps = int(cap.get(cv2.CAP_PROP_FPS))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # videoWriter = cv2.VideoWriter(os.path.join(save_dir, vid + "_test.avi"), cv2.VideoWriter_fourcc('I', '4', '2', '0'),
    #                               fps, size)
    videoWriter = cv2.VideoWriter(os.path.join(save_dir, vid + "_test.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    idx = 0
    while success:
        t1 = time()
        idx = idx + 1
        print("Test", idx, "image")
        image_id = "%012d" % idx
        video_floder = os.path.join(save_dir, "vp_results")
        color_floder = os.path.join(save_dir, "color_results")

        image = frame  # hpanet need bgr
        # image = cv2.imread(os.path.join(IMAGE_DIR, vid, image_id) + '.jpg')
        # Run detection
        # results = model.detect([image[:, :, ::-1]])
        t2 = time()
        results = model.detect([image], pre_images=[], pre_masks=[], pre_parts=[])
        t3 = time()
        print("1, model test one image:", t3 - t2, "s")
        # Visualize results
        r = results[0]
        # visualize.vis_insts(image, color_floder, image_id, r['boxes'], r['masks'], r['class_ids'], r['scores'])
        masked_image = visualize.vis_insts_opencv(frame, color_floder, image_id, r['boxes'], r['masks'],
                                                  r['class_ids'], r['scores'])
        # masked_image = visualize.vis_insts_opencv(image, color_floder, image_id, r['boxes'], r['masks'], r['class_ids'],
        #                            r['scores'])
        t4 = time()
        # print("vis_insts", t3 - t2)
        global_parsing_map, color_map, part_inst_maps = visualize.write_inst_part_result(video_floder, color_floder,
                                                                                         image.shape[0], image.shape[1],
                                                                                         image_id, r['boxes'],
                                                                                         r['masks'], r['scores'],
                                                                                         r['global_parsing'])
        vis_global_image = cv2.addWeighted(masked_image, 1, global_parsing_map, 0.4, 0)
        vis_global_path = os.path.join(save_dir, "vis", "vis_global_%s.png" % image_id)
        if not os.path.exists(os.path.dirname(vis_global_path)):
            os.makedirs(os.path.dirname(vis_global_path))
        cv2.imwrite(vis_global_path, vis_global_image)
        videoWriter.write(vis_global_image)
        print("    write_inst_part_result", time() - t4, "s")
        print("2, visualize results", time() - t2, "s")
        print("3, test and visualize one image:", time() - t1, "s")
        success, frame = cap.read()
    cap.release()
    print("total", time() - t0, "s")
    cap.release()


if __name__ == '__main__':
    main()
