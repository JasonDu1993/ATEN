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
from models.parsing_rcnn_model_dilated_se import PARSING_RCNN
# from models.parsing_rcnn_model_dilated import PARSING_RCNN


class trainConfig(ParsingRCNNModelConfig):
    NAME = "vip_singleframe_20190908c"
    # NAME = "vip_singleframe_test"
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
PRETRAIN_MODEL_PATH = os.path.join(ROOT_DIR, "checkpoints", "parsing_rcnn.h5")
# PRETRAIN_MODEL_PATH = "/home/sk49/workspace/zhoudu/ATEN/outputs/vip_singleframe_20190326a/checkpoints/" \
#                       "parsing_rcnn_vip_singleframe_20190326a_epoch038_loss0.491_valloss0.550.h5"

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = "./outputs"
DEFAULT_DATASET_DIR = "/home/sk49/workspace/dataset/VIP"
# DEFAULT_DATASET_DIR = "D:\dataset\VIP_tiny"

############################################################
#  Training
############################################################


if __name__ == '__main__':
    """command:
    nohup python3 train_parsingrcnn.py >> outs/train_vip_video_20190903a.out &
    tail -f outs/train_vip_video_20190903a.out
    """
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
    dataset_train.load_vip(args.dataset, "train")
    dataset_train.prepare()
    # dataset_train = VIPDataset()
    # dataset_train.load_vip(args.dataset, "traintiny")
    # dataset_train.prepare()

    # Validation dataset
    dataset_val = VIPDataset()
    dataset_val.load_vip(args.dataset, "val")
    dataset_val.prepare()
    # dataset_val = VIPDataset()
    # dataset_val.load_vip(args.dataset, "traintiny")
    # dataset_val.prepare()

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
