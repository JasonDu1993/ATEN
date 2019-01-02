import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys

sys.path.insert(0, os.getcwd())
import skimage.io

from configs.vip import ParsingRCNNModelConfig

from models.parsing_rcnn_model import PARSING_RCNN
from utils import visualize

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "outputs")

# Path to trained weights file
# Download this file and place in the root of your 
# project (See README file for details)
DATASET_DIR = "/home/sk49/workspace/dataset/VIP"
MODEL_PATH = "./checkpoints/parsing_rcnn.h5"
# Directory of images to run detection on
IMAGE_DIR = DATASET_DIR + "/Images"
IMAGE_LIST = DATASET_DIR + "/lists/test_id.txt"

RES_DIR = "./vis/vip_test_singleframe"
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
    if os.path.exists(os.path.join(video_floder, 'instance_part', image_id) + '.png'):
        continue
    image = skimage.io.imread(os.path.join(IMAGE_DIR, vid, image_id) + '.jpg')
    # Run detection
    results = model.detect([image])

    # Visualize results
    r = results[0]
    visualize.vis_insts(image, video_floder, image_id, r['rois'], r['masks'], r['class_ids'], r['scores'])
    visualize.write_inst_part_result(video_floder, image.shape[0], image.shape[1], image_id, r['rois'], r['masks'],
                                     r['scores'], r['global_parsing'])
