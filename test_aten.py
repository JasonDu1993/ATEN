import os
import sys
from time import time

t0 = time()
sys.path.insert(0, os.getcwd())
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
DATASET_DIR = "/home/sk49/workspace/dataset/VIP"

# Root directory of the project
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "outputs_aten")
# Directory of images to run detection on
# MODEL_PATH = "./checkpoints/aten_p2l3.h5"
# MODEL_PATH = "./outputs_aten/vip_video_20190103va/checkpoints/" \
#              "aten_vip_video_20190103va_epoch200_loss1.441_valloss1.354.h5"
MODEL_PATH = "./outputs_aten/vip_video_20190103vma/checkpoints/" \
             "aten_vip_video_20190103vma_epoch200_loss3.426_valloss3.532.h5"
IMAGE_DIR = DATASET_DIR + "/Images"
FRONT_FRAME_LIST_DIR = DATASET_DIR + "/front_frame_list"
BEHIND_FRAME_LIST_DIR = DATASET_DIR + "/behind_frame_list"
mode = "val"
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
    image_id = image_ids[i]
    image_info = dataset.image_info[image_id]
    file_line = image_info['id']
    ind = file_line.rfind('/')
    vid = file_line[:ind]
    im_name = file_line[ind + 1:]
    path = os.path.join(RES_DIR, vid)
    if not os.path.exists(path):
        os.makedirs(path)

    if os.path.exists(os.path.join(path, 'global_parsing', '%s.png' % im_name)):
        continue
    print(i, file_line)
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
    visualize.vis_insts(cur_frame, path, im_name, r['rois'], r['masks'], r['class_ids'], r['scores'])
    visualize.write_inst_part_result(path, cur_frame.shape[0], cur_frame.shape[1], im_name,
                                     r['rois'], r['masks'], r['scores'], r['global_parsing'])
    print("aten visualize results", time() - t2, "s")

print("total", time() - t0, "s")
