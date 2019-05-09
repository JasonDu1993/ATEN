# -*- coding: utf-8 -*-
# @Time    : 2019/5/9 14:55
# @Author  : Jason
# @Email   : 1358681631@qq.com
# @File    : test_triple_model_multiprocessing.py
# @Software: PyCharm
import os
import sys
import cv2
import math
import time
import numpy as np
from multiprocessing import Queue, Process
from tqdm import tqdm
import skimage.io
import matplotlib
from configs import vip
from utils import visualize

sys.path.insert(0, os.getcwd())
matplotlib.use('Agg')


class InferenceConfig(vip.VideoModelConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    KEY_RANGE_L = 3
    PROCESS_COUNT = 1
    RECURRENT_UNIT = "gru"


DATASET_DIR = "/home/sk49/workspace/dataset/VIP"

# Root directory of the project
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "outputs_aten")
# Directory of images to run detection on
# MODEL_PATH = "./checkpoints/aten_p2l3.h5"
# MODEL_PATH = "./outputs_aten/vip_video_20190103va/checkpoints/" \
#              "aten_vip_video_20190103va_epoch200_loss1.441_valloss1.354.h5"
MODEL_PATH = "/home/sk49/workspace/zhoudu/ATEN/outputs_aten/vip_video_20190507va/checkpoints/" \
             "aten_vip_video_20190507va_epoch035_loss0.597_valloss0.547.h5"
IMAGE_DIR = DATASET_DIR + "/Images"
FRONT_FRAME_LIST_DIR = DATASET_DIR + "/front_frame_list"
BEHIND_FRAME_LIST_DIR = DATASET_DIR + "/behind_frame_list"
IMAGE_LIST = DATASET_DIR + "/lists/test_id.txt"
mode = "test"
# RES_DIR = "./vis_aten/test_vip_video_20190103va_epoch169"
# RES_DIR = "./vis_aten/test_vip_video_20190507va_epoch035"
RES_DIR = "./vis_aten/debug"

if not os.path.exists(RES_DIR):
    os.makedirs(RES_DIR)


def worker(image_ids, dataset, infer_config):
    """
    Args:
        image_ids : 输入数据，为图片的id信息，for example: video106/000000000001
        dataset:
        infer_config:
    """
    # gpu数量
    t0 = time.time()
    import tensorflow as tf
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    import keras.backend as K
    config_tf = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    config_tf.gpu_options.per_process_gpu_memory_fraction = 0.3
    session = tf.Session(config=config_tf)
    from models.aten_triplemodel_dilated import ATEN_PARSING_RCNN
    model = ATEN_PARSING_RCNN(mode='inference', config=infer_config, model_dir=MODEL_DIR)
    model.load_weights(MODEL_PATH, by_name=True)

    c = 0
    print("load model time:", time.time() - t0, "s", len(image_ids), type(image_ids))
    for i in range(len(image_ids)):
        c += 1
        t1 = time.time()
        image_id = image_ids[i]
        image_info = dataset.image_info[image_id]
        file_line = image_info['id']
        ind = file_line.rfind('/')
        vid = file_line[:ind]
        im_name = file_line[ind + 1:]
        video_floder = os.path.join(RES_DIR, "vp_results", vid)
        color_floder = os.path.join(RES_DIR, "color_results", vid)
        if not os.path.exists(video_floder):
            os.makedirs(video_floder)
        if not os.path.exists(color_floder):
            os.makedirs(color_floder)
        # print(os.path.join(video_floder, "global_parsing", im_name + ".png"))
        p1 = os.path.exists(os.path.join(video_floder, "global_parsing", im_name + ".png"))
        p2 = os.path.exists(os.path.join(video_floder, "instance_parsing", im_name + ".png"))
        p3 = os.path.exists(os.path.join(video_floder, "instance_parsing", im_name + ".txt"))
        p4 = os.path.exists(os.path.join(video_floder, "instance_segmentation", im_name + ".png"))
        p5 = os.path.exists(os.path.join(video_floder, "instance_segmentation", im_name + ".txt"))
        p6 = os.path.exists(os.path.join(color_floder, "color", "global_" + im_name + ".png"))
        p7 = os.path.exists(os.path.join(color_floder, "color", "inst_" + im_name + ".png"))
        p8 = os.path.exists(os.path.join(color_floder, "color", "vis_" + im_name + ".png"))
        # print(p1, p2, p3, p4, p5, p6, p7, p8)
        if p1 and p2 and p3 and p4 and p5 and p6 and p7 and p8:
            continue
        print("line", c, file_line)
        cur_frame = dataset.load_image(image_id)
        keys, identity_ind = dataset.load_infer_keys(image_id, infer_config.KEY_RANGE_L, 3)
        assert len(keys) == 3, "keys num must be 3"
        key1 = keys[0]
        key2 = keys[1]
        key3 = keys[2]

        t2 = time.time()
        r = model.detect([cur_frame, ], [key1, ], [key2, ], [key3, ], [identity_ind, ])[0]
        t3 = time.time()
        print("  (1), model test one image", t3 - t2, "s")
        # print("detect out ", r['class_ids'].shape[0], "person")
        # visualize.vis_insts(cur_frame, video_floder, im_name, r['rois'], r['masks'], r['class_ids'], r['scores'])
        visualize.vis_insts_opencv(cur_frame[:, :, ::-1], color_floder, im_name, r['rois'], r['masks'], r['class_ids'],
                                   r['scores'])
        t4 = time.time()
        visualize.write_inst_part_result(video_floder, color_floder, cur_frame.shape[0], cur_frame.shape[1], im_name,
                                         r['rois'], r['masks'], r['scores'], r['global_parsing'])
        print("    write_inst_part_result", time.time() - t4, "s")
        print("  (2), visualize results", time.time() - t3, "s")
        print("  (3), test and visualize results", time.time() - t1, "s")

    print("total", time.time() - t0, "s")
    session.close()


def multiprocess_main():
    start = time.time()
    infer_config = InferenceConfig()
    num_workers = infer_config.PROCESS_COUNT
    # 总数据量
    dataset = vip.VIPDataset()
    dataset.load_vip(DATASET_DIR, mode)
    dataset.prepare()
    image_ids = np.copy(dataset.image_ids)
    image_num = len(image_ids)
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
        split_data = image_ids[start:end]
        print("split_data", split_data)
        # 各个进程开始
        proc = Process(target=worker, args=(split_data, dataset, infer_config))
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
