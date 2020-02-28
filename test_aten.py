import os
import sys
from time import time

t0 = time()
sys.path.insert(0, os.getcwd())
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from utils.flowlib import *
from configs import vip

from models.aten_model import ATEN_PARSING_RCNN
from utils import visualize


class InferenceConfig(vip.VideoModelConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    KEY_RANGE_L = 3
    RECURRENT_UNIT = "gru"


config = InferenceConfig()
# DATASET_DIR = "/home/sk49/workspace/dataset/VIP"
DATASET_DIR = "D:\dataset\VIP_tiny"

# Root directory of the project
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "outputs_aten")
# Directory of images to run detection on
MODEL_PATH = "./checkpoints/aten_p2l3.h5"
# MODEL_PATH = "./outputs_aten/vip_video_20190103va/checkpoints/" \
#              "aten_vip_video_20190103va_epoch200_loss1.441_valloss1.354.h5"
# MODEL_PATH = "./outputs_aten/vip_video_20190103vma/checkpoints/" \
#              "aten_vip_video_20190103vma_epoch200_loss3.426_valloss3.532.h5"
IMAGE_DIR = DATASET_DIR + "/Images"
FRONT_FRAME_LIST_DIR = DATASET_DIR + "/front_frame_list"
BEHIND_FRAME_LIST_DIR = DATASET_DIR + "/behind_frame_list"
mode = "traintiny"
# RES_DIR = "./vis_aten/test_vip_video_20190103va_epoch169"
RES_DIR = "./vis_aten/" + mode + "_vip_video_20190103vma_epoch200"
# RES_DIR = "./vis_aten/val_vip_video"

if not os.path.exists(RES_DIR):
    os.makedirs(RES_DIR)
config.display()

# Create model object in inference mode.

model = ATEN_PARSING_RCNN(mode='inference', config=config, model_dir=MODEL_DIR)
model.load_weights(MODEL_PATH, by_name=True)
dataset = vip.VIPDataset()

dataset.load_vip(DATASET_DIR, mode)
dataset.prepare()
image_ids = np.copy(dataset.image_ids)
c = 0
print("load model time:", time() - t0, "s")
for i in range(len(image_ids)):
    c += 1
    image_id = image_ids[i]
    image_info = dataset.image_info[image_id]
    file_line = image_info['id']
    ind = file_line.rfind('/')
    vid = file_line[:ind]
    im_name = file_line[ind + 1:]
    video_floder = os.path.join(RES_DIR, "vp_results", vid)
    color_floder = os.path.join(RES_DIR, "color_results", vid)
    if not os.path.exists(video_floder):
        os.makedirs(video_floder)
    if not os.path.exists(color_floder):
        os.makedirs(color_floder)

    # print(os.path.join(video_floder, "global_parsing", im_name + ".png"))
    p1 = os.path.exists(os.path.join(video_floder, "global_parsing", im_name + ".png"))
    p2 = os.path.exists(os.path.join(video_floder, "instance_parsing", im_name + ".png"))
    p3 = os.path.exists(os.path.join(video_floder, "instance_parsing", im_name + ".txt"))
    p4 = os.path.exists(os.path.join(video_floder, "instance_segmentation", im_name + ".png"))
    p5 = os.path.exists(os.path.join(video_floder, "instance_segmentation", im_name + ".txt"))
    p6 = os.path.exists(os.path.join(color_floder, "color", "global_" + im_name + ".png"))
    p7 = os.path.exists(os.path.join(color_floder, "color", "inst_" + im_name + ".png"))
    p8 = os.path.exists(os.path.join(color_floder, "color", "vis_" + im_name + ".png"))
    # print(p1, p2, p3, p4, p5, p6, p7, p8)
    if p1 and p2 and p3 and p4 and p5 and p6 and p7 and p8:
        continue
    print("line", c, file_line)
    cur_frame = dataset.load_image(image_id)
    keys, identity_ind = dataset.load_infer_keys(image_id, config.KEY_RANGE_L, 3)
    assert len(keys) == 3, "keys num must be 3"
    key1 = keys[0]
    key2 = keys[1]
    key3 = keys[2]

    t1 = time()
    r = model.detect([cur_frame, ], [key1, ], [key2, ], [key3, ], [identity_ind, ])[0]
    t2 = time()
    print("aten test one image", t2 - t1, "s")
    # print("detect out ", r['class_ids'].shape[0], "person")
    visualize.vis_insts(cur_frame, video_floder, im_name, r['boxes'], r['masks'], r['class_ids'], r['scores'])
    global_parsing_map, color_map, part_inst_maps = visualize.write_inst_part_result(video_floder, color_floder,
                                                                                     cur_frame.shape[0],
                                                                                     cur_frame.shape[1], im_name,
                                                                                     r['boxes'], r['masks'],
                                                                                     r['scores'], r['global_parsing'])
    print("aten visualize results", time() - t2, "s")

print("total", time() - t0, "s")
