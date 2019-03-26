import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
import sys

sys.path.insert(0, os.getcwd())
import skimage.io

from configs.vip import ParsingRCNNModelConfig

# from models.parsing_rcnn_model import PARSING_RCNN
from models.parsing_rcnn_model_dilated import PARSING_RCNN
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
DATASET_DIR = "/home/sk49/workspace/dataset/VIP"
MODEL_PATH = "./outputs/vip_singleframe_20181229a/checkpoints/parsing_rcnn_vip_singleframe_20181229a_epoch043.h5"
# MODEL_PATH = "./outputs/vip_singleframe_20181229ma/checkpoints/parsing_rcnn_vip_singleframe_20181229ma_epoch086.h5"
# MODEL_PATH = "./outputs/vip_singleframe_test/checkpoints/parsing_rcnn_vip_singleframe_test_epoch001.h5"
# MODEL_PATH = "./checkpoints/parsing_rcnn.h5"
# Directory of images to run detection on
IMAGE_DIR = DATASET_DIR + "/Images"
IMAGE_LIST = DATASET_DIR + "/lists/test_id.txt"

# RES_DIR = "./vis/test_vip_singleframe_20181229ma_epoch086"
RES_DIR = "./vis/test1"
if not os.path.exists(RES_DIR):
    os.makedirs(RES_DIR)


class InferenceConfig(ParsingRCNNModelConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = PARSING_RCNN(mode="inference", config=config, model_dir=MODEL_DIR)

# Load weights trained on MS-COCO
model.load_weights(MODEL_PATH, by_name=True)
print("load model", time() - t0, "s")
c = 0
rfp = open(IMAGE_LIST, 'r')
for line in rfp.readlines():
    line = line.strip()
    c += 1
    print(c, line)
    ind = line.find('/')
    vid = line[:ind]
    image_id = line[ind + 1:]
    video_floder = os.path.join(RES_DIR, vid)
    if not os.path.exists(video_floder):
        os.makedirs(video_floder)
    if os.path.exists(os.path.join(video_floder, 'global_parsing', image_id) + '.png'):
        continue
    image = skimage.io.imread(os.path.join(IMAGE_DIR, vid, image_id) + '.jpg')
    # Run detection
    t1 = time()
    results = model.detect([image])
    t2 = time()
    print("test one image", t2 - t1, "s")

    # Visualize results
    r = results[0]
    visualize.vis_insts(image, video_floder, image_id, r['rois'], r['masks'], r['class_ids'], r['scores'])
    visualize.write_inst_part_result(video_floder, image.shape[0], image.shape[1], image_id, r['rois'], r['masks'],
                                     r['scores'], r['global_parsing'])
    print("visualize results", time() - t2, "s")

print("total", time() - t0, "s")
