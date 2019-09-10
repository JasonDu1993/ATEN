import os
import tensorflow as tf

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# config = tf.ConfigProto()
# # config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
# session = tf.Session(config=config)
import sys
import cv2

sys.path.insert(0, os.getcwd())
import skimage.io

import matplotlib

matplotlib.use('Agg')
from configs.vip import ParsingRCNNModelConfig

from models.parsing_rcnn_model import PARSING_RCNN
# from models.parsing_rcnn_model_miouloss import PARSING_RCNN
from utils import visualize
from time import time

t0 = time()
# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "outputs")

# Path to trained weights file
# Download this file and place in the root of your 
# project (See README file for details)
# DATASET_DIR = "/home/sk49/workspace/dataset/VIP"
DATASET_DIR = "D:\dataset\VIP_tiny"
# MODEL_PATH = "/home/sk49/workspace/zhoudu/ATEN/outputs/vip_singleframe_20190408a/checkpoints/" \
#              "parsing_rcnn_vip_singleframe_20190408a_epoch073_loss0.401_valloss0.391.h5"
# MODEL_PATH = "./outputs/vip_singleframe_20181229ma/checkpoints/parsing_rcnn_vip_singleframe_20181229ma_epoch086.h5"
# MODEL_PATH = "./outputs/vip_singleframe_test/checkpoints/parsing_rcnn_vip_singleframe_test_epoch001_loss0.585_valloss0.497.h5"
MODEL_PATH = "./checkpoints/parsing_rcnn.h5"
# Directory of images to run detection on
IMAGE_DIR = DATASET_DIR + "/Images"
IMAGE_LIST = DATASET_DIR + "/lists/traintiny_id.txt"
# IMAGE_LIST = DATASET_DIR + "/lists/trainval_id.txt"

# RES_DIR = "./vis/trainval_vip_singleframe_20190408a_epoch073000"
# RES_DIR = "./vis/test_vip_singleframe_20190326a_epoch032_t"
RES_DIR = "./vis/viptiny_test"
flag = False
if not os.path.exists(RES_DIR):
    os.makedirs(RES_DIR)
    flag = True


# else:
#     print(RES_DIR, "测试文件已存在，请检查是否需要修改预测文件名")

class InferenceConfig(ParsingRCNNModelConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def main():
    global config
    config = InferenceConfig()
    config.display()
    # Create model object in inference mode.
    model = PARSING_RCNN(mode="inference", config=config, model_dir=MODEL_DIR)
    # Load weights trained on MS-COCO
    s0 = time()
    model.load_weights(MODEL_PATH, by_name=True)
    print("load model", time() - s0, "s")
    c = 0
    rfp = open(IMAGE_LIST, 'r')
    for line in rfp.readlines():
        t1 = time()
        line = line.strip()
        c += 1
        ind = line.find('/')
        vid = line[:ind]
        image_id = line[ind + 1:]
        video_floder = os.path.join(RES_DIR, "vp_results", vid)
        color_floder = os.path.join(RES_DIR, "color_results", vid)
        if not os.path.exists(video_floder):
            os.makedirs(video_floder)
        if not os.path.exists(color_floder):
            os.makedirs(color_floder)
        p1 = os.path.exists(os.path.join(video_floder, "global_parsing", image_id + ".png"))
        p2 = os.path.exists(os.path.join(video_floder, "instance_parsing", image_id + ".png"))
        p3 = os.path.exists(os.path.join(video_floder, "instance_parsing", image_id + ".txt"))
        p4 = os.path.exists(os.path.join(video_floder, "instance_segmentation", image_id + ".png"))
        p5 = os.path.exists(os.path.join(video_floder, "instance_segmentation", image_id + ".txt"))
        p6 = os.path.exists(os.path.join(color_floder, "color", "global_" + image_id + ".png"))
        p7 = os.path.exists(os.path.join(color_floder, "color", "inst_" + image_id + ".png"))
        p8 = os.path.exists(os.path.join(color_floder, "color", "vis_" + image_id + ".png"))
        # print(p1, p2, p3, p4, p5, p6, p7, p8)
        if p1 and p2 and p3 and p4 and p5 and p6 and p7 and p8:
            continue
        print("line", c, line)
        image = skimage.io.imread(os.path.join(IMAGE_DIR, vid, image_id) + '.jpg')
        # image = cv2.imread(os.path.join(IMAGE_DIR, vid, image_id) + '.jpg')
        # Run detection
        # results = model.detect([image[:, :, ::-1]])
        t2 = time()
        results = model.detect([image])
        t3 = time()
        print("  1, model test one image:", t3 - t2, "s")
        # Visualize results
        r = results[0]
        # masked_image = visualize.vis_insts(image, color_floder, image_id, r['rois'], r['masks'], r['class_ids'], r['scores'])
        masked_image = visualize.vis_insts_opencv(image[:, :, ::-1], color_floder, image_id, r['rois'], r['masks'],
                                                  r['class_ids'], r['scores'])
        # masked_image = visualize.vis_insts_opencv(image, color_floder, image_id, r['rois'], r['masks'], r['class_ids'],
        #                            r['scores'])
        t4 = time()
        print("    (1)vis_insts:", t4 - t3)
        global_parsing_map, color_map = visualize.write_inst_part_result(video_floder, color_floder, image.shape[0],
                                                                         image.shape[1], image_id, r['rois'],
                                                                         r['masks'], r['scores'], r['global_parsing'])
        vis_global_image = cv2.addWeighted(masked_image, 1, global_parsing_map, 0.4, 0)
        cv2.imwrite(os.path.join(color_floder, "color", "vis_global_%s.png" % image_id), vis_global_image)
        print("    (2)write_inst_part_result(A and B total time):", time() - t4, "s")
        print("  2, visualize results total time:", time() - t3, "s")
        print("  3, test and visualize one image:", time() - t1, "s")
    print("total", time() - t0, "s")


if __name__ == '__main__':
    main()
