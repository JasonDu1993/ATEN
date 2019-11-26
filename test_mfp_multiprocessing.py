# -*- coding: utf-8 -*-
# @Time    : 2019/11/13 10:36
# @Author  : Jason
# @Email   : 1358681631@qq.com
# @File    : test_mfp_multiprocessing.py
# @Software: PyCharm
import os
import sys
import cv2
import math
import time
import platform
from multiprocessing import Queue, Process
import matplotlib

from utils import visualize
from utils.util_load_mfp_data import get_scale, load_pre_image_names, load_pre_image_datas, load_pre_image_boxes
import tensorflow as tf
import importlib

# modified 1
name = "models.mfp_model_roiprebox_tinyinput_rpn_premaskpart"
module = importlib.import_module(name)
sys.path.insert(0, os.getcwd())

matplotlib.use('Agg')
MACHINE_NAME = platform.node()
# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "outputs")


class InferenceConfig(module.MFPConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    # modified 5
    PROCESS_NAME = "mfp_20191125a_epoch016"  # for tmp tested image name
    GPU_COUNT = 1  # only 1
    # modified 6
    PROCESS_COUNT = 3
    IMAGES_PER_GPU = 1  # only 1
    BATCH_SIZE = 1  # only 1
    # whether save the predicted visualized image
    ISCOLOR = True
    # open image tool
    ISOPENCV = True


# Path to trained weights file
# Download this file and place in the root of your
# project (See README file for details)

# linux
if MACHINE_NAME == "Jason":
    # win
    DATASET_DIR = r"D:\dataset\VIP_tiny"
    # modified 2
    MODEL_PATH = "outputs/mfp_20191116a/checkpoints/parsing_rcnn_mfp_20191116a_epoch003_loss0.741_valloss0.753.h5"
    # modified 3
    RES_DIR = "./vis_mfp/val_mfp_20191116a_epoch003"
    # modified 4
    gpus = ["1"]
    IMAGE_DIR = DATASET_DIR + "/Images"
    IMAGE_LIST = DATASET_DIR + "/lists/traintiny_id.txt"
    PRE_IMAGE_DIR = r"D:\dataset\VIP_tiny"
    PRE_PREDICT_DATA_DIR = r"D:\dataset\VIP_tiny"
else:
    DATASET_DIR = "/home/sk49/workspace/dataset/VIP"
    # modified 2
    MODEL_PATH = "/home/sk49/workspace/zhoudu/ATEN/outputs/mfp_20191125a/checkpoints" + "/" + \
                 "parsing_rcnn_mfp_20191125a_epoch016_loss0.681_valloss0.639.h5"
    # modified 3
    RES_DIR = "./vis_mfp/val_mfp_20191125a_epoch016"
    # modified 4
    gpus = ["1"]
    IMAGE_DIR = DATASET_DIR + "/Images"
    IMAGE_LIST = DATASET_DIR + "/lists/val_id.txt"
    PRE_IMAGE_DIR = r"/home/sk49/workspace/dataset/VIP"
    PRE_PREDICT_DATA_DIR = r"/home/sk49/workspace/zhoudu/ATEN/vis/origin_val_vip_singleframe_parsing_rcnn"

    # test all val image in VIP dataset
    # IMAGE_DIR = DATASET_DIR + "/videos/val_videos_frames"
    # IMAGE_LIST = DATASET_DIR + "/lists/val_all_frames_id.txt"

flag = False
if not os.path.exists(RES_DIR):
    os.makedirs(RES_DIR)
    flag = True


def worker(images, infer_config, gpu_id, tested_images_set, tested_path):
    """
    Args:
        images : 输入数据，为图片的id信息，for example: video106/000000000001
        infer_config:
        gpu_id: str
        tested_images_set:
        tested_images_set: str, record the tested image name(video_name/image_id), using for continuing testing when interrupt
    """
    # gpu数量
    t0 = time.time()
    iscolor = infer_config.ISCOLOR
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    images_set = set(images)
    l = len(images_set)
    images_set = images_set - tested_images_set
    f_tested = open(tested_path, "a")
    tf_config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.3
    session = tf.Session(config=tf_config)
    # The below line need to correct every test time
    # from models.mfp_model_roiprebox_tinyinput_rpn import MFP
    if infer_config is None:
        infer_config = InferenceConfig()
    model = module.MFP(mode="inference", config=infer_config, model_dir=MODEL_DIR)
    # Load weights trained on MS-COCO
    s0 = time.time()
    model.load_weights(MODEL_PATH, by_name=True)
    print("load model", time.time() - s0, "s")
    # time.sleep(10)
    c = l - len(images_set)
    images_set = sorted(images_set)
    for line in images_set:
        t1 = time.time()
        line = line.strip()
        c += 1

        ind = line.find('/')
        vid = line[:ind]
        image_id = line[ind + 1:]
        video_floder = os.path.join(RES_DIR, "vp_results", vid)
        color_floder = os.path.join(RES_DIR, "color_results", vid)

        print("line", c, line, "pid:", os.getpid())
        image = cv2.imread(os.path.join(IMAGE_DIR, vid, image_id) + '.jpg')
        # scale = get_scale(image.shape, infer_config)
        pre_image_names = load_pre_image_names(line, key_num=infer_config.PRE_MULTI_FRAMES)
        if len(pre_image_names) == 0:
            continue
        pre_images, pre_masks, pre_parts = load_pre_image_datas(line, pre_image_names, infer_config,
                                                                PRE_IMAGE_DIR, PRE_PREDICT_DATA_DIR)
        # pre_boxes = load_pre_image_boxes(pre_image_names, scale, PRE_PREDICT_DATA_DIR)
        # Run detection
        t2 = time.time()
        results = model.detect([image], pre_images, pre_masks, pre_parts, isopencv=infer_config.ISOPENCV)
        t3 = time.time()
        print("  1, model test one image:", t3 - t2, "s")
        # Visualize results
        r = results[0]
        if iscolor:
            # visualize.vis_insts(image, color_floder, image_id, r['boxes'], r['masks'], r['class_ids'], r['scores'])
            masked_image = visualize.vis_insts_opencv(image, color_floder, image_id, r['boxes'], r['masks'],
                                                      r['class_ids'], r['scores'])
            # masked_image = visualize.vis_insts_opencv(image, color_floder, image_id, r['boxes'], r['masks'],
            #                       r['class_ids'], r['scores'])
        t4 = time.time()
        print("  2 (1)vis_insts:", t4 - t3)
        global_parsing_map, color_map = visualize.write_inst_part_result(video_floder, color_floder, image.shape[0],
                                                                         image.shape[1], image_id, r['boxes'],
                                                                         r['masks'], r['scores'], r['global_parsing'],
                                                                         iscolor=iscolor)
        if iscolor:
            vis_global_image = cv2.addWeighted(masked_image, 1, global_parsing_map, 0.4, 0)
            cv2.imwrite(os.path.join(color_floder, "color", "vis_global_%s.png" % image_id), vis_global_image)
        print("    (2)write_inst_part_result:", time.time() - t4, "s")
        print("    (3)visualize results total time:", time.time() - t3, "s")
        print("  3, test and visualize one image:", time.time() - t1, "s")
        f_tested.write(line + "\n")
        f_tested.flush()
    f_tested.close()
    print("total", time.time() - t0, "s")
    session.close()


def multiprocess_main():
    start = time.time()
    infer_config = InferenceConfig()
    num_workers = infer_config.PROCESS_COUNT
    # 总数据量
    with open(IMAGE_LIST, 'r') as f:
        images_list = f.readlines()
        image_num = len(images_list)
        image_split = math.ceil(image_num / (num_workers * len(gpus)))
        # 所有进程
        procs = []
        # 已经测试过的
        tested_images_set = set()
        for p in os.listdir(RES_DIR):
            if p.startswith("tested_"):
                with open(os.path.join(RES_DIR, p), "r") as f:
                    for l in f.readlines():
                        tested_images_set.add(l)
        print("test image num:", image_num, "and tested image num:", len(tested_images_set), "leave num:",
              image_num - len(tested_images_set))
        # 对于每个进程
        for i, gpu_id in enumerate(gpus):
            for j in range(num_workers):
                # 数据分块
                start = (i * num_workers + j) * image_split
                end = min(start + image_split, image_num)
                split_data = images_list[start:end]
                # 各个进程开始
                tested_path = os.path.join(RES_DIR,
                                           "tested_" + infer_config.PROCESS_NAME + "_gpu" + str(i) + "_proc" + str(
                                               j) + ".txt")
                proc = Process(target=worker,
                               args=(split_data, infer_config, gpu_id, tested_images_set, tested_path))
                proc.start()
                print('process:%d, start:%d, end:%d. tested_path: %s' % (proc.pid, start, end, tested_path))
                procs.append(proc)
                # # 数据量，将queue中数据取出
                # for i in range(image_num):
                #     ret = main_queue.get()
                #     # 将ret写入file_out
                #     print('{}'.format(ret), file=file_out)
                #     # 进度条更新
                #     pbar.update(1)

    for p in procs:
        p.join()
        endtime = time.time()
        print("process:" + p.name + " pid:" + str(p.pid) + " time:", endtime - start, "s")


if __name__ == '__main__':
    from time import strftime

    print("testing multi process at:", strftime("%Y_%m%d_%H%M%S"))
    t0 = time.time()
    multiprocess_main()
    print("MAIN END!")
    print("MAIN TOTAL TIME:", time.time() - t0, "s")
