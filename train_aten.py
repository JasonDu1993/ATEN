import os
import sys
from time import time

sys.path.insert(0, os.getcwd())
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
import numpy as np

from configs.vip import VideoModelConfig
from configs.vip import VIPDataset
from models import aten_model_dilated as modellib


class trainConfig(VideoModelConfig):
    # NAME = "vip_video_20190103va"
    NAME = "debug"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # STEPS_PER_EPOCH = 3000
    STEPS_PER_EPOCH = 20
    # VALIDATION_STEPS = 100
    VALIDATION_STEPS = 10
    SAVE_MODEL_PERIOD = 1
    # Weight decay regularization
    WEIGHT_DECAY = 0.0001
    # Image mean (RGB)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])
    KEY_RANGE_L = 3
    RECURRENT_UNIT = 'gru'


# Root directory of the project
ROOT_DIR = os.getcwd()
# Path to trained weights file
PRETRAIN_MODEL_PATH = os.path.join(ROOT_DIR, "checkpoints", "aten_p2l3.h5")
PARSING_RCNN_MODEL_PATH = os.path.join(ROOT_DIR, "checkpoints", "parsing_rcnn.h5")
FLOWNET_MODEL_PATH = os.path.join(ROOT_DIR, "checkpoints", "flownet2-S.h5")
# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = "./outputs_aten"
DEFAULT_DATASET_DIR = "/home/sk49/workspace/dataset/VIP"
############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    t0 = time()
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on Pascal Person Part.')
    parser.add_argument('--dataset', required=False,
                        default=DEFAULT_DATASET_DIR,
                        metavar="/path/to/coco/",
                        help='Directory of the dataset')
    parser.add_argument('--model', required=False,
                        default="pretrain",
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')

    args = parser.parse_args()
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    config = trainConfig()
    config.display()

    # Create model
    model = modellib.ATEN_PARSING_RCNN(mode="training", config=config, model_dir=args.logs)
    # Select weights file to load
    if args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()[1]
    elif args.model.lower() == "pretrain":
        model_path = PRETRAIN_MODEL_PATH
    else:
        model_path = args.model

    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True)

    # Training dataset. Use the training set and 35K from the
    # validation set, as as in the Mask RCNN paper.
    dataset_train = VIPDataset()
    dataset_train.load_vip(args.dataset, "trainval")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = VIPDataset()
    dataset_val.load_vip(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***

    # Fine tune all layers
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=0.001,
                epochs=200,
                layers='all',
                period=config.SAVE_MODEL_PERIOD)
    print("total", time() - t0, "s")
