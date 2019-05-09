# -*- coding: utf-8 -*-
# @Time    : 2019/5/6 14:59
# @Author  : Jason
# @Email   : 1358681631@qq.com
# @File    : vip_triple_model.py.py
# @Software: PyCharm
import os
import random

import cv2
import numpy as np
import skimage.io
import skimage.transform
from utils.util import Dataset


class VIPDatasetForTripleModel(Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def add_parsing_class(self, source, class_id, class_name):
        # Does the class exist already?
        for info in self.parsing_class_info:
            if info['source'] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.parsing_class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })

    def load_vip(self, dataset_dir, subset):
        # Add classes
        self.add_class("VIP", 1, "person")  # self.class_info.append
        self.add_parsing_class("VIP", 1, "hat")  # self.parsing_class_info.append
        self.add_parsing_class("VIP", 2, "hair")
        self.add_parsing_class("VIP", 3, "gloves")
        self.add_parsing_class("VIP", 4, "sun-glasses")
        self.add_parsing_class("VIP", 5, "upper-clothes")
        self.add_parsing_class("VIP", 6, "dress")
        self.add_parsing_class("VIP", 7, "coat")
        self.add_parsing_class("VIP", 8, "socks")
        self.add_parsing_class("VIP", 9, "pants")
        self.add_parsing_class("VIP", 10, "torso-skin")
        self.add_parsing_class("VIP", 11, "scarf")
        self.add_parsing_class("VIP", 12, "skirt")
        self.add_parsing_class("VIP", 13, "face")
        self.add_parsing_class("VIP", 14, "left-arm")
        self.add_parsing_class("VIP", 15, "right-arm")
        self.add_parsing_class("VIP", 16, "left-leg")
        self.add_parsing_class("VIP", 17, "right-leg")
        self.add_parsing_class("VIP", 18, "left-shoe")
        self.add_parsing_class("VIP", 19, "right-shoe")

        # Path
        image_dir = os.path.join(dataset_dir, 'Images')
        # Add images
        # Generate random specifications of images (i.e. color and
        # list of shapes sizes and locations). This is more compact than
        # actual images. Images are generated on the fly in load_image().
        rfp = open(os.path.join(dataset_dir, 'lists', '%s_id.txt' % subset),
                   'r')  # '/home/sk49/workspace/dataset/VIP/lists/trainval_id.txt'
        for line in rfp.readlines():
            image_id = line.strip()
            self.add_image("VIP", image_id=image_id,
                           path=os.path.join(image_dir, '%s.jpg' % image_id),
                           front_frame_list=os.path.join(dataset_dir, 'front_frame_list', '%s.txt' % image_id),
                           behind_frame_list=os.path.join(dataset_dir, 'behind_frame_list', '%s.txt' % image_id),
                           inst_anno=os.path.join(dataset_dir, 'Human_ids', '%s.png' % image_id),
                           part_anno=os.path.join(dataset_dir, 'Category_ids', '%s.png' % image_id),
                           part_rev_anno=os.path.join(dataset_dir, 'Category_rev_ids', '%s.png' % image_id))

    def load_keys(self, image_id, key_range, key_num):
        image_info = self.image_info[image_id]
        with open(image_info['front_frame_list'], 'r') as fp:
            front_frame_ids = [x.strip() for x in fp.readlines()]
        with open(image_info['behind_frame_list'], 'r') as fp:
            behind_frame_ids = [x.strip() for x in fp.readlines()]

        # get image_dir
        image_dir = image_info['path']
        image_dir = image_dir[:image_dir.rfind('.')]
        adjacent_frames_dir = image_dir.replace("Images", "adjacent_frames")
        frame_ind = int(image_dir.split('/')[-1])  # 301

        if len(front_frame_ids) > 0:
            most_left_ind = int(front_frame_ids[-1])  # 291
        else:
            most_left_ind = frame_ind
        if len(behind_frame_ids) > 0:
            most_right_ind = int(behind_frame_ids[-1])  # 311
        else:
            most_right_ind = frame_ind
        # print(most_left_ind, frame_ind, most_right_ind)
        new_key_left = max(most_left_ind, frame_ind - int(key_range // 2))  # 300
        new_key_right = min(most_right_ind, frame_ind + (key_range - 1 - int(key_range // 2)))  # 302
        sel_pool = []  # frame_ind = 301, such as <class 'list'>: [[300, 1], [301, 0], [302, 1]].
        for i in range(new_key_left, new_key_right + 1):
            if i != frame_ind:
                sel_pool.append([i, 1])
            else:
                sel_pool.append([i, 0])
        sel = random.choice(sel_pool)
        new_key_ind = sel[0]
        identity_ind = sel[1]
        # print("new key ind", new_key_ind)
        keys = []
        most_left_key_ind = new_key_ind - key_range * (key_num - 1)  # 296
        most_right_key_ind = new_key_ind + key_range * (key_num - 1)  # 308
        get_left = True
        # print(most_left_key_ind, most_right_key_ind)
        if random.randint(0, 1):
            # if <most_left_ind, get right keys
            if most_left_key_ind < most_left_ind:
                get_left = False
                if most_right_key_ind > most_right_ind:
                    raise Exception("There is not enough adjacent frames, right_key %d VS right_limit %d" % (
                        most_right_key_ind, most_right_ind))
            else:
                get_left = True
        else:
            # if >most_right_ind, get left keys
            if most_right_key_ind > most_right_ind:
                get_left = True
                if most_left_key_ind < most_left_ind:
                    raise Exception("There is not enough adjacent frames left_key %d VS left_limit %d" % (
                        most_left_key_ind, most_left_ind))
            else:
                get_left = False
        if get_left:
            if identity_ind == 0:
                tmp_path = image_info['path']
            else:
                tmp_path = '%s/%012d.jpg' % (adjacent_frames_dir, new_key_ind)
            tmp_key = skimage.io.imread(tmp_path)
            keys.append(tmp_key)
            for i in range(1, key_num):
                tmp_ind = new_key_ind - i * key_range
                tmp_path = '%s/%012d.jpg' % (adjacent_frames_dir, tmp_ind)
                tmp_key = skimage.io.imread(tmp_path)
                keys.append(tmp_key)
        else:
            if identity_ind == 0:
                tmp_path = image_info['path']
            else:
                tmp_path = '%s/%012d.jpg' % (adjacent_frames_dir, new_key_ind)
            tmp_key = skimage.io.imread(tmp_path)
            keys.append(tmp_key)
            for i in range(1, key_num):
                tmp_ind = new_key_ind + i * key_range
                tmp_path = '%s/%012d.jpg' % (adjacent_frames_dir, tmp_ind)
                tmp_key = skimage.io.imread(tmp_path)
                keys.append(tmp_key)

        return keys, identity_ind

    def load_infer_keys(self, image_id, key_range, key_num):
        image_info = self.image_info[image_id]
        with open(image_info['front_frame_list'], 'r') as fp:
            front_frame_ids = [x.strip() for x in fp.readlines()]
        with open(image_info['behind_frame_list'], 'r') as fp:
            behind_frame_ids = [x.strip() for x in fp.readlines()]

        # get image_dir
        image_dir = image_info['path']
        image_dir = image_dir[:image_dir.rfind('.')]
        adjacent_frames_dir = image_dir.replace("Images", "adjacent_frames")
        frame_ind = int(image_dir.split('/')[-1])

        if len(front_frame_ids) > 0:
            most_left_ind = int(front_frame_ids[-1])
        else:
            most_left_ind = frame_ind
        if len(behind_frame_ids) > 0:
            most_right_ind = int(behind_frame_ids[-1])
        else:
            most_right_ind = frame_ind
        # print(most_left_ind, frame_ind, most_right_ind)
        new_key_left = max(most_left_ind, frame_ind - int(key_range // 2))
        new_key_right = min(most_right_ind, frame_ind + (key_range - 1 - int(key_range // 2)))
        sel_pool = []
        for i in range(new_key_left, new_key_right + 1):
            if i != frame_ind:
                sel_pool.append([i, 1])
            else:
                sel_pool.append([i, 0])
        sel = random.choice(sel_pool)
        new_key_ind = sel[0]
        identity_ind = sel[1]
        keys = []
        most_left_key_ind = new_key_ind - key_range * (key_num - 1)
        most_right_key_ind = new_key_ind + key_range * (key_num - 1)
        get_left = True

        # if < most_left_ind, get right keys
        if most_left_key_ind < most_left_ind:
            get_left = False
            if most_right_key_ind > most_right_ind:
                raise Exception("There is not enough adjacent frames, right_key %d VS right_limit %d" % (
                    most_right_key_ind, most_right_ind))
        else:
            get_left = True

        if get_left:
            if identity_ind == 0:
                tmp_path = image_info['path']
            else:
                tmp_path = '%s/%012d.jpg' % (adjacent_frames_dir, new_key_ind)
            tmp_key = skimage.io.imread(tmp_path)
            keys.append(tmp_key)
            for i in range(1, key_num):
                tmp_ind = new_key_ind - i * key_range
                tmp_path = '%s/%012d.jpg' % (adjacent_frames_dir, tmp_ind)
                tmp_key = skimage.io.imread(tmp_path)
                keys.append(tmp_key)
        else:
            if identity_ind == 0:
                tmp_path = image_info['path']
            else:
                tmp_path = '%s/%012d.jpg' % (adjacent_frames_dir, new_key_ind)
            tmp_key = skimage.io.imread(tmp_path)
            keys.append(tmp_key)
            for i in range(1, key_num):
                tmp_ind = new_key_ind + i * key_range
                tmp_path = '%s/%012d.jpg' % (adjacent_frames_dir, tmp_ind)
                tmp_key = skimage.io.imread(tmp_path)
                keys.append(tmp_key)

        return keys, identity_ind

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        if image_info["source"] != "VIP":
            return super(VIPDatasetForTripleModel, self).load_mask(image_id)

        class_ids = []
        instance_masks = []

        gt_inst_data = cv2.imread(image_info["inst_anno"],
                                  cv2.IMREAD_GRAYSCALE)  # gt_inst_data shape: (720, 1280) min:0 max:person_num uint8
        # img_path="'/home/sk49/workspace/dataset/VIP/Human_ids/videos88/000000001676.png'"
        # gt_inst_data = cv2.imread(image_info["inst_anno"])
        # gt_inst_data = cv2.imread(img_path)
        # get all instance label list
        unique_inst = np.unique(gt_inst_data)  # 去除重复元素，并按元素由大到小返回一个新的无元素重复的元组或者列表 [0 1 2 3 4 5]
        background_ind = np.where(unique_inst == 0)[0]
        unique_inst = np.delete(unique_inst, background_ind)  # 去除代表背景的0， [1 2 3 4 5]

        # print("unique_inst", unique_inst)

        for i in range(unique_inst.shape[0]):
            inst_mask = unique_inst[i]
            # get specific instance mask
            im_mask = (gt_inst_data == inst_mask)
            instance_masks.append(im_mask)
            class_ids.append(1)
        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(VIPDatasetForTripleModel, self).load_mask(image_id)

    def load_part(self, image_id):
        """Load part category map for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a category_id map [height, width].

        Returns:
        parts: A uint8 array of shape [height, width].
        """
        image_info = self.image_info[image_id]
        gt_part_data = cv2.imread(image_info["part_anno"],
                                  cv2.IMREAD_GRAYSCALE)  # shape: (720, 1280) min:0 max: part_num
        return gt_part_data.astype(np.uint8)

    def load_reverse_part(self, image_id):
        """Load part category map for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a category_id map [height, width].

        Returns:
        parts: A uint8 array of shape [height, width].
        """
        image_info = self.image_info[image_id]
        gt_part_data = cv2.imread(image_info["part_rev_anno"], cv2.IMREAD_GRAYSCALE)
        return gt_part_data.astype(np.uint8)


