import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
session = tf.Session(config=config)
import sys
import cv2

sys.path.insert(0, os.getcwd())
import numpy as np

from configs.vipdataset_for_mfp import ParsingRCNNModelConfig
from models.mfp_model_roiprebox_tinyinput import MFP
from utils import visualize
from utils.util_load_mfp_data import get_scale, load_pre_image_names, load_pre_image_datas, load_pre_image_boxes
from time import time

t0 = time()
# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "outputs")

# Path to trained weights file
# Download this file and place in the root of your 
# project (See README file for details)
DATASET_DIR = "/home/sk49/workspace/dataset/VIP"
MODEL_PATH = "/home/sk49/workspace/zhoudu/ATEN/outputs/mfp_20191112c/checkpoints" + "/" + \
             "parsing_rcnn_mfp_20191112c_epoch017_loss0.505_valloss0.511.h5"
IMAGE_DIR = DATASET_DIR + "/Images"
IMAGE_LIST = DATASET_DIR + "/lists/val_id.txt"
PRE_IMAGE_DIR = r"/home/sk49/workspace/dataset/VIP"
PRE_PREDICT_DATA_DIR = r"/home/sk49/workspace/zhoudu/ATEN/vis/origin_val_vip_singleframe_parsing_rcnn"
RES_DIR = "./vis_mfp/val_mfp_20191112c_epoch017"

# DATASET_DIR = "D:\dataset\VIP_tiny"
# MODEL_PATH = "outputs/mfp_20191112b/checkpoints/parsing_rcnn_mfp_20191112b_epoch015_loss0.497_valloss0.506.h5"
# IMAGE_DIR = DATASET_DIR + "/Images"
# IMAGE_LIST = DATASET_DIR + "/lists/traintiny_id.txt"
# PRE_IMAGE_DIR = r"D:\dataset\VIP_tiny"
# PRE_PREDICT_DATA_DIR = r"D:\dataset\VIP_tiny"
# RES_DIR = "./vis_mfp/debug"

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

    BATCH_SIZE = 1
    # whether save the predicted visualized image
    ISCOLOR = True
    # open image tool
    ISOPENCV = True

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 450  # 450, 256
    IMAGE_MAX_DIM = 512  # 512, 416， 384（16*24）
    # use small pre image for training
    PRE_IMAGE_SHAPE = [128, 128, 3]  # needed 128(PRE_IMAGE_SHAPE[0]) * 4 = 512(IMAGE_MAX_DIM)

    PRE_MULTI_FRAMES = 3
    RECURRENT_UNIT = "gru"
    assert RECURRENT_UNIT in ["gru", "lstm"]
    RECURRENT_FILTER = 64
    USE_RPN_ROIS = True  #


def main():
    global config
    config = InferenceConfig()
    config.display()
    # Create model object in inference mode.
    model = MFP(mode="inference", config=config, model_dir=MODEL_DIR)
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
        print("line", c, line)
        image = cv2.imread(os.path.join(IMAGE_DIR, vid, image_id) + '.jpg')
        scale = get_scale(image.shape, config)
        pre_image_names = load_pre_image_names(line, key_num=config.PRE_MULTI_FRAMES)
        if len(pre_image_names) == 0:
            continue
        pre_images, pre_masks, pre_parts= load_pre_image_datas(line, pre_image_names, config, PRE_IMAGE_DIR,
                                                                       PRE_PREDICT_DATA_DIR)
        pre_boxes = load_pre_image_boxes(pre_image_names, scale, PRE_PREDICT_DATA_DIR)
        # Run detection
        t2 = time()
        results = model.detect([image], pre_images, pre_masks, pre_parts, pre_boxes, isopencv=config.ISOPENCV)
        t3 = time()
        print("  1, model test one image:", t3 - t2, "s")
        # Visualize results
        r = results[0]
        masked_image = visualize.vis_insts_opencv(image, color_floder, image_id, r['boxes'], r['masks'],
                                                  r['class_ids'], r['scores'])
        t4 = time()
        print("    (1)vis_insts:", t4 - t3)
        global_parsing_map, color_map = visualize.write_inst_part_result(video_floder, color_floder, image.shape[0],
                                                                         image.shape[1], image_id, r['boxes'],
                                                                         r['masks'], r['scores'], r['global_parsing'])
        vis_global_image = cv2.addWeighted(masked_image, 1, global_parsing_map, 0.4, 0)
        cv2.imwrite(os.path.join(color_floder, "color", "vis_global_%s.png" % image_id), vis_global_image)
        print("    (2)write_inst_part_result(A and B total time):", time() - t4, "s")
        print("  2, visualize results total time:", time() - t3, "s")
        print("  3, test and visualize one image:", time() - t1, "s")
    print("total", time() - t0, "s")


if __name__ == '__main__':
    main()
