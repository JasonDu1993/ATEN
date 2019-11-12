import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
session = tf.Session(config=config)
import sys
import cv2

sys.path.insert(0, os.getcwd())
import numpy as np

from configs.vipdataset_for_mfp import ParsingRCNNModelConfig
from utils.util import resize_image, resize_mask, resize_part_mfp
from models.mfp_model import MFP
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
# MODEL_PATH = "/home/sk49/workspace/zhoudu/ATEN/outputs/mfp_20191031a/checkpoints" + "/" + \
#              "parsing_rcnn_mfp_20191031a_epoch023_loss0.899_valloss0.571.h5"
# IMAGE_DIR = DATASET_DIR + "/Images"
# IMAGE_LIST = DATASET_DIR + "/lists/val_id.txt"
# PRE_IMAGE_DIR = r"/home/sk49/workspace/dataset/VIP"
# PRE_PREDICT_DATA_DIR = r"/home/sk49/workspace/zhoudu/ATEN/vis/origin_val_vip_singleframe_parsing_rcnn"

DATASET_DIR = "D:\dataset\VIP_tiny"
MODEL_PATH = "outputs/mfp_20191028b/checkpoints/parsing_rcnn_mfp_20191028b_epoch003_loss1.366_valloss1.006.h5"
IMAGE_DIR = DATASET_DIR + "/Images"
IMAGE_LIST = DATASET_DIR + "/lists/traintiny_id.txt"
PRE_IMAGE_DIR = r"D:\dataset\VIP_tiny"
PRE_PREDICT_DATA_DIR = r"D:\dataset\VIP_tiny"

# RES_DIR = "./vis/trainval_vip_singleframe_20190408a_epoch073000"
# RES_DIR = "./vis/test_vip_singleframe_20190326a_epoch032_t"
RES_DIR = "./vis_mfp/debug"
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
    ISCOLOR = True
    ISOPENCV = True

    IMAGE_MIN_DIM = 256  # 450, 256
    IMAGE_MAX_DIM = 384  # 512, 416， 384（16*24）
    PRE_MULTI_FRAMES = 3
    RECURRENT_UNIT = "gru"
    assert RECURRENT_UNIT in ["gru", "lstm"]
    RECURRENT_FILTER = 64


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


def load_pre_image_datas(image_name, pre_image_names, config):
    """

    Args:
        image_name: image_name includes video_name/image_id, for example videos45/000000000176
        pre_image_names: list, the value is list, len=2, which represent video_name, pre_image_id
        config: The model config object

    Returns:
        pre_images: list, the value is numpy.ndarray, shape [resize_height=512, resize_width=512, 3(BGR)]
        pre_masks: list, the value is numpy.ndarray, shape [resize_height, resize_width, 1],
            which value include 0 ~ num_person, 0 is bg, 1 ~ num_person is the person label
        pre_parts: list, the value is numpy.ndarray, shape [resize_height, resize_width, num_class=20],
            which value include 0 ~ 19, 0 is bg, 1 ~ 19 is the person part label
        scale:

    """
    video_name, image_id = image_name.strip().split("/")
    pre_images = []
    pre_masks = []
    pre_parts = []
    scale = 1
    for pre_video_name, pre_image_id in pre_image_names:
        pre_image_path = os.path.join(PRE_IMAGE_DIR, "adjacent_frames", pre_video_name, image_id, pre_image_id + ".jpg")
        pre_mask_path = os.path.join(PRE_PREDICT_DATA_DIR, "vp_results", pre_video_name, "instance_segmentation",
                                     pre_image_id + ".png")
        pre_part_path = os.path.join(PRE_PREDICT_DATA_DIR, "vp_results", pre_video_name, "global_parsing",
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
        pre_image, window, scale, padding = resize_image(pre_image, max_dim=config.IMAGE_MAX_DIM,
                                                         padding=config.IMAGE_PADDING, isopencv=True)
        pre_mask = resize_mask(pre_mask, scale, padding, isopencv=True)[:, :, np.newaxis]  # shape [512, 512,1]
        pre_part = resize_part_mfp(pre_part, scale, padding, isopencv=True)  # [512,512,20]
        pre_images.append(pre_image[np.newaxis, ...])
        pre_masks.append(pre_mask[np.newaxis, ...])
        pre_parts.append(pre_part[np.newaxis, ...])
    return pre_images, pre_masks, pre_parts, scale


def load_pre_image_boxes(pre_image_names, scale):
    """

    Args:
        pre_image_names: list, the value is list, len=2, which represent video_name, pre_image_id
        scale: The scale factor used to resize the image

    Returns:
        pre_boxes: list, the value is list, len is 4, which represent y1, x1, y2, x2.

    """
    pre_boxes = []
    for pre_video_name, pre_image_id in pre_image_names:
        boxes_path = os.path.join(PRE_PREDICT_DATA_DIR, "vp_results", pre_video_name, "instance_segmentation",
                                  pre_image_id + ".txt")
        with open(boxes_path, "r") as f:
            for line in f.readlines():
                y1, x1, y2, x2 = list(map(int, line.strip().split(" ")[1:]))
                pre_boxes.append([round(y1 * scale), round(x1 * scale), round(y2 * scale), round(x2 * scale)])
    return pre_boxes


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
        pre_image_names = load_pre_image_names(line, key_num=config.PRE_MULTI_FRAMES)
        if len(pre_image_names) == 0:
            continue
        pre_images, pre_masks, pre_parts, scale = load_pre_image_datas(line, pre_image_names, config)
        pre_boxes = load_pre_image_boxes(pre_image_names, scale)
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
