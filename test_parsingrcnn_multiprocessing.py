# -*- coding: utf-8 -*-
# @Time    : 2019/5/7 15:00
# @Author  : Jason
# @Email   : 1358681631@qq.com
# @File    : test_parsingrcnn_multiprocessing.py
# @Software: PyCharm
import os
import sys
import cv2
import math
import time
import platform
import importlib
from multiprocessing import Queue, Process
from tqdm import tqdm
import skimage.io
import matplotlib

from configs.vip import ParsingRCNNModelConfig
from utils import visualize

import tensorflow as tf

sys.path.insert(0, os.getcwd())

matplotlib.use('Agg')
MACHINE_NAME = platform.node()
# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "outputs")

# Path to trained weights file
# Download this file and place in the root of your
# project (See README file for details)
# modified 1
name = "models.parsing_rcnn_model"
module = importlib.import_module(name)


class InferenceConfig(ParsingRCNNModelConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    # modified 2
    PROCESS_NAME = "val_vip_singleframe_20191126a_epoch029"  # for process name
    GPU_COUNT = 1
    PROCESS_COUNT = 3
    IMAGES_PER_GPU = 1
    BATCH_SIZE = 1
    ISCOLOR = True


# linux
if MACHINE_NAME == "Jason":
    # win
    DATASET_DIR = r"D:\dataset\VIP_tiny"
    # modified 3
    MODEL_PATH = "./checkpoints/parsing_rcnn.h5"
    # modified 4
    RES_DIR = "./vis/debug"
    # modified 5
    gpus = ["2"]
    IMAGE_DIR = DATASET_DIR + "/Images"
    IMAGE_LIST = DATASET_DIR + "/lists/traintiny_id.txt"
else:
    DATASET_DIR = "/home/sk49/workspace/dataset/VIP"
    # modified 3
    MODEL_PATH = "/home/sk49/workspace/zhoudu/ATEN/outputs/vip_singleframe_20191126a/checkpoints" + "/" + \
                 "parsing_rcnn_vip_singleframe_20191126a_epoch029_loss4.181_valloss4.490.h5"
    # modified 4
    RES_DIR = "./vis/val_vip_singleframe_20191126a_epoch029"
    # modified 5
    gpus = ["1"]
    IMAGE_DIR = DATASET_DIR + "/Images"
    IMAGE_LIST = DATASET_DIR + "/lists/val_id.txt"

    # test all val image in VIP dataset
    # IMAGE_DIR = DATASET_DIR + "/videos/val_videos_frames"
    # IMAGE_LIST = DATASET_DIR + "/lists/val_all_frames_id.txt"

flag = False
if not os.path.exists(RES_DIR):
    os.makedirs(RES_DIR)
    flag = True


# else:
#     print(RES_DIR, "测试文件已存在，请检查是否需要修改预测文件名")


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
    if infer_config is None:
        infer_config = InferenceConfig()
    model = module.PARSING_RCNN(mode="inference", config=infer_config, model_dir=MODEL_DIR)
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
        image = skimage.io.imread(os.path.join(IMAGE_DIR, vid, image_id) + '.jpg')
        # image = cv2.imread(os.path.join(IMAGE_DIR, vid, image_id) + '.jpg')
        # Run detection
        # results = model.detect([image[:, :, ::-1]])
        t2 = time.time()
        results = model.detect([image])
        t3 = time.time()
        print("  1, model test one image:", t3 - t2, "s")
        # Visualize results
        r = results[0]
        if iscolor:
            # visualize.vis_insts(image, color_floder, image_id, r['boxes'], r['masks'], r['class_ids'], r['scores'])
            masked_image = visualize.vis_insts_opencv(image[:, :, ::-1], color_floder, image_id, r['boxes'], r['masks'],
                                                      r['class_ids'], r['scores'])
            # masked_image = visualize.vis_insts_opencv(image, color_floder, image_id, r['boxes'], r['masks'],
            #                       r['class_ids'], r['scores'])
        t4 = time.time()
        print("  2 (1)vis_insts:", t4 - t3)
        global_parsing_map, color_map, part_inst_maps = visualize.write_inst_part_result(video_floder, color_floder,
                                                                                         image.shape[0], image.shape[1],
                                                                                         image_id, r['boxes'],
                                                                                         r['masks'], r['scores'],
                                                                                         r['global_parsing'],
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
