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
from multiprocessing import Queue, Process
from tqdm import tqdm
import skimage.io
import matplotlib

from configs.vip import ParsingRCNNModelConfig
# from models.parsing_rcnn_model import PARSING_RCNN
from utils import visualize

import tensorflow as tf

sys.path.insert(0, os.getcwd())

matplotlib.use('Agg')

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "outputs")

# Path to trained weights file
# Download this file and place in the root of your
# project (See README file for details)
DATASET_DIR = "/home/sk49/workspace/dataset/VIP"
MODEL_PATH = "/home/sk49/workspace/zhoudu/ATEN/outputs/vip_singleframe_20190520a/checkpoints" + "/" + \
             "parsing_rcnn_vip_singleframe_20190520a_epoch036_loss0.552_valloss0.540.h5"
RES_DIR = "./vis/test_vip_singleframe_20190520a_epoch036"
# RES_DIR = "./vis/debug"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Directory of images to run detection on
IMAGE_DIR = DATASET_DIR + "/Images"
IMAGE_LIST = DATASET_DIR + "/lists/test_id.txt"

# IMAGE_DIR = DATASET_DIR + "/videos/train_videos_frames"
# IMAGE_LIST = DATASET_DIR + "/lists/train_all_frames_id_part3.txt"


flag = False
if not os.path.exists(RES_DIR):
    os.makedirs(RES_DIR)
    flag = True


# else:
#     print(RES_DIR, "测试文件已存在，请检查是否需要修改预测文件名")

class InferenceConfig(ParsingRCNNModelConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    PROCESS_COUNT = 3
    IMAGES_PER_GPU = 1


def worker(images, infer_config):
    """
    Args:
        images : 输入数据，为图片的id信息，for example: video106/000000000001
        infer_config:
    """
    # gpu数量
    t0 = time.time()
    import keras.backend as K
    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    session = tf.Session(config=config)
    # from models.parsing_rcnn_model_resfpn_dilated_se import PARSING_RCNN
    from models.parsing_rcnn_model_resfpn_dilated_se_attention2 import PARSING_RCNN
    if infer_config is None:
        infer_config = InferenceConfig()
    model = PARSING_RCNN(mode="inference", config=infer_config, model_dir=MODEL_DIR)
    # Load weights trained on MS-COCO
    s0 = time.time()
    model.load_weights(MODEL_PATH, by_name=True)
    print("load model", time.time() - s0, "s")
    # time.sleep(10)
    c = 0
    for line in images:
        t1 = time.time()
        line = line.strip()
        c += 1

        ind = line.find('/')
        vid = line[:ind]
        image_id = line[ind + 1:]
        video_floder = os.path.join(RES_DIR, "vp_results", vid)
        color_floder = os.path.join(RES_DIR, "color_results", vid)
        # print(os.path.join(video_floder, "global_parsing", image_id + ".png"))
        p1 = os.path.exists(os.path.join(video_floder, "global_parsing", image_id + ".png"))
        p2 = os.path.exists(os.path.join(video_floder, "instance_parsing", image_id + ".png"))
        p3 = os.path.exists(os.path.join(video_floder, "instance_parsing", image_id + ".txt"))
        p4 = os.path.exists(os.path.join(video_floder, "instance_segmentation", image_id + ".png"))
        p5 = os.path.exists(os.path.join(video_floder, "instance_segmentation", image_id + ".txt"))
        p6 = os.path.exists(os.path.join(color_floder, "color", "global_" + image_id + ".png"))
        p7 = os.path.exists(os.path.join(color_floder, "color", "inst_" + image_id + ".png"))
        p8 = os.path.exists(os.path.join(color_floder, "color", "vis_" + image_id + ".png"))
        # print(p1, p2, p3, p4, p5, p6, p7, p8)
        if p1 and p2 and p3 and p4 and p5 and p6 and p7 and p8:
            continue
        print("line", c, line)
        image = skimage.io.imread(os.path.join(IMAGE_DIR, vid, image_id) + '.jpg')
        # image = cv2.imread(os.path.join(IMAGE_DIR, vid, image_id) + '.jpg')
        # Run detection
        # results = model.detect([image[:, :, ::-1]])
        t2 = time.time()
        results = model.detect([image])
        t3 = time.time()
        print("  (1), model test one image:", t3 - t2, "s")
        # Visualize results
        r = results[0]
        # visualize.vis_insts(image, color_floder, image_id, r['rois'], r['masks'], r['class_ids'], r['scores'])
        masked_image = visualize.vis_insts_opencv(image[:, :, ::-1], color_floder, image_id, r['rois'], r['masks'],
                                                  r['class_ids'], r['scores'])
        # masked_image = visualize.vis_insts_opencv(image, color_floder, image_id, r['rois'], r['masks'], r['class_ids'],
        #                            r['scores'])
        t4 = time.time()
        # print("vis_insts", t3 - t2)
        global_parsing_map, color_map = visualize.write_inst_part_result(video_floder, color_floder, image.shape[0],
                                                                         image.shape[1], image_id, r['rois'],
                                                                         r['masks'], r['scores'], r['global_parsing'])
        vis_global_image = cv2.addWeighted(masked_image, 1, global_parsing_map, 0.4, 0)
        cv2.imwrite(os.path.join(color_floder, "color", "vis_global_%s.png" % image_id), vis_global_image)
        # vis_inst_image = cv2.addWeighted(masked_image, 1, color_map, 0.4, 0)
        # cv2.imwrite(os.path.join(color_floder, "color", "vis_ins_%s.png" % image_id), vis_inst_image)
        print("    write_inst_part_result", time.time() - t4, "s")
        print("  (2), visualize results", time.time() - t2, "s")
        print("  (3), test and visualize one image:", time.time() - t1, "s")
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
        print("test image num:", image_num)
        image_split = math.ceil(image_num / num_workers)
        # 所有进程
        procs = []
        # 队列缓存

        # 对于每个进程
        for i in range(num_workers):
            # 数据分块
            start = i * image_split
            end = min(start + image_split, image_num)
            split_data = images_list[start:end]
            # 各个进程开始
            proc = Process(target=worker, args=(split_data, infer_config))
            print('process:%d, start:%d, end:%d' % (i, start, end))
            proc.start()
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
    t0 = time.time()
    multiprocess_main()
    print("MAIN END!")
    print("MAIN TOTAL TIME:", time.time() - t0, "s")
