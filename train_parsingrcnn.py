import os
import sys
from time import time

sys.path.insert(0, os.getcwd())
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from configs.vip import ParsingRCNNModelConfig
from configs.vip import VIPDataset
# from models.parsing_rcnn_model import PARSING_RCNN
from models.parsing_rcnn_model_resstage5_dilated_se import PARSING_RCNN


class trainConfig(ParsingRCNNModelConfig):
    NAME = "vip_singleframe_20190515a"
    # NAME = "vip_singleframe_test"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4
    STEPS_PER_EPOCH = 2000
    # STEPS_PER_EPOCH = 20
    VALIDATION_STEPS = 100
    # VALIDATION_STEPS = 10
    SAVE_MODEL_PERIOD = 1
    KEY_RANGE_L = 3


# Root directory of the project
ROOT_DIR = os.getcwd()

# Path to trained weights file
# PRETRAIN_MODEL_PATH = os.path.join(ROOT_DIR, "checkpoints", "parsing_rcnn.h5")
PRETRAIN_MODEL_PATH = "/home/sk49/workspace/zhoudu/ATEN/outputs/vip_singleframe_20190513a/checkpoints" + "/" + \
                      "parsing_rcnn_vip_singleframe_20190513a_epoch047_loss0.363_valloss0.329.h5"

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = "./outputs"
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
    model = PARSING_RCNN(mode="training", config=config,
                         model_dir=args.logs)

    # Select weights file to load
    if args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()[1]
    elif args.model.lower() == "pretrain":
        model_path = PRETRAIN_MODEL_PATH
    else:
        model_path = args.model
    # common load weight 
    print("Loading weights ", model_path)
    t0 = time()
    model.load_weights(model_path, by_name=True)
    print("Loaded weights ", time() - t0, "s")
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

    model.train(dataset_train, dataset_val,
                learning_rate=0.001,
                epochs=200,
                layers='all',
                period=config.SAVE_MODEL_PERIOD)

    # model.train(dataset_train, dataset_val,
    #             learning_rate=0.0001,
    #             epochs=150,
    #             layers='all',
    #             period=config.SAVE_MODEL_PERIOD)
    print("total", (time() - t0), "s")
