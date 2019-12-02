import os
import sys
import platform
from time import time

import tensorflow as tf

MACHINE_NAME = platform.node()
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, os.getcwd())

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from configs.vipdataset_for_mfp import VIPDatasetForMFP
from models.mfp_resfpn_c5d_edgamf256_e357_part357_partse_part import MFPNet, MFPConfig


class trainConfig(MFPConfig):
    NAME = "mfp_20191202c"
    # NAME = "mfp_debug"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4
    # IMAGES_PER_GPU = 1
    STEPS_PER_EPOCH = 2000
    # STEPS_PER_EPOCH = 2
    VALIDATION_STEPS = 100
    # VALIDATION_STEPS = 1
    SAVE_MODEL_PERIOD = 1


# Root directory of the project
ROOT_DIR = os.getcwd()

# Path to trained weights file


# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = "./outputs"
# linux
if MACHINE_NAME == "Jason":
    # win
    PRETRAIN_MODEL_PATH = os.path.join(ROOT_DIR, "checkpoints", "parsing_rcnn.h5")
    DEFAULT_DATASET_DIR = "D:\dataset\VIP_tiny"
    pre_image_train_dir = "D:\dataset\VIP_tiny"
    pre_image_val_dir = "D:\dataset\VIP_tiny"
else:
    # PRETRAIN_MODEL_PATH = "/home/sk49/workspace/zhoudu/ATEN/outputs/mfp_20191028b/checkpoints" + "/" + \
    #                       "parsing_rcnn_mfp_20191028b_epoch003_loss1.366_valloss1.006.h5"
    # PRETRAIN_MODEL_PATH = os.path.join(ROOT_DIR, "checkpoints", "parsing_rcnn.h5")
    PRETRAIN_MODEL_PATH = os.path.join(ROOT_DIR, "checkpoints", "hpa_20191130d_epoch019.h5")
    DEFAULT_DATASET_DIR = "/home/sk49/workspace/dataset/VIP"
    pre_image_train_dir = "/home/sk49/workspace/zhoudu/ATEN/vis/origin_train_vip_singleframe_parsing_rcnn"
    pre_image_val_dir = "/home/sk49/workspace/zhoudu/ATEN/vis/origin_val_vip_singleframe_parsing_rcnn"

############################################################
#  Training
############################################################


if __name__ == '__main__':
    """command:
    nohup python3 train_mfp.py >> outs/train_vip_video_20190903a.out &
    tail -f outs/train_vip_video_20190903a.out
    """
    import argparse
    from time import strftime

    print("training at:", strftime("%Y_%m%d_%H%M%S"))
    t0 = time()
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on Pascal Person Part.')
    parser.add_argument('--dataset', required=False,
                        default=DEFAULT_DATASET_DIR,
                        metavar="/path/to/dataset/",
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
    model = MFPNet(mode="training", config=config, model_dir=args.logs)

    # Select weights file to load
    # if args.model.lower() == "last":
    #     # Find last trained weights
    #     model_path = model.find_last()[1]
    # elif args.model.lower() == "pretrain":
    #     model_path = PRETRAIN_MODEL_PATH
    # else:
    #     model_path = args.model
    # common load weight
    model_path = PRETRAIN_MODEL_PATH
    print("Loading weights ", model_path)
    t0 = time()
    if model_path:
        model.load_weights(model_path, by_name=True)
        print("Loaded weights ", time() - t0, "s")
    # Training dataset. Use the training set and 35K from the
    # validation set, as as in the Mask RCNN paper.
    if MACHINE_NAME == "Jason":
        dataset_train = VIPDatasetForMFP()
        dataset_train.load_vip(args.dataset, "traintiny", pre_image_train_dir)
        dataset_train.prepare()
        # Validation dataset
        dataset_val = VIPDatasetForMFP()
        dataset_val.load_vip(args.dataset, "traintiny", pre_image_val_dir)
        dataset_val.prepare()
    else:
        dataset_train = VIPDatasetForMFP()
        dataset_train.load_vip(args.dataset, "train", pre_image_train_dir)
        dataset_train.prepare()
        # Validation dataset
        dataset_val = VIPDatasetForMFP()
        dataset_val.load_vip(args.dataset, "val", pre_image_val_dir)
        dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***

    # Fine tune all layers

    model.train(dataset_train, dataset_val,
                learning_rate=0.0001,
                epochs=200,
                layers='all',
                period=config.SAVE_MODEL_PERIOD)

    # model.train(dataset_train, dataset_val,
    #             learning_rate=0.0001,
    #             epochs=150,
    #             layers='all',
    #             period=config.SAVE_MODEL_PERIOD)
    print("total", (time() - t0), "s")
