"""
Mask R-CNN
Display and Visualization Functions.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""
import os
import random
import itertools
import colorsys
import numpy as np
from skimage.measure import find_contours
import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon
import IPython.display
import cv2
import utils
from time import time
import scipy.io as sio

matplotlib.use('Agg')


############################################################
#  Visualization
############################################################

def display_images(images, titles=None, cols=4, cmap=None, norm=None,
                   interpolation=None):
    """Display the given set of images, optionally with titles.
    images: list or array of image tensors in HWC format.
    titles: optional. A list of titles to display with each image.
    cols: number of images per row
    cmap: Optional. Color map to use. For example, "Blues".
    norm: Optional. A Normalize instance to map values to colors.
    interpolation: Optional. Image interporlation to use for display.
    """
    titles = titles if titles is not None else [""] * len(images)
    rows = len(images) // cols + 1
    plt.figure(figsize=(14, 14 * rows // cols))
    i = 1
    for image, title in zip(images, titles):
        plt.subplot(rows, cols, i)
        plt.title(title, fontsize=9)
        plt.axis('off')
        plt.imshow(image.astype(np.uint8), cmap=cmap,
                   norm=norm, interpolation=interpolation)
        i += 1
    plt.show()


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def random_colors_opencv(N=20):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    # colors = [list((matplotlib.colors.hsv_to_rgb([x, 1.0, 1.0]) * 255).astype(int)) for x in
    #           np.arange(0, 1, 1.0 / N)]
    # np.random.shuffle(colors)
    # 20个颜色通过上述代码获取
    if N <= 50:
        colors = [[255, 0, 30], [0, 224, 255], [0, 255, 132], [112, 255, 0], [255, 0, 0], [255, 0, 183], [0, 255, 71],
                  [0, 163, 255], [0, 40, 255], [142, 255, 0], [173, 255, 0], [112, 0, 255], [255, 0, 244],
                  [255, 0, 122], [20, 255, 0], [0, 132, 255], [142, 0, 255], [204, 0, 255], [0, 255, 193],
                  [0, 255, 102], [0, 71, 255], [173, 0, 255], [255, 30, 0], [255, 0, 214], [51, 0, 255], [203, 255, 0],
                  [0, 255, 40], [0, 255, 163], [81, 255, 0], [255, 214, 0], [234, 0, 255], [20, 0, 255], [255, 61, 0],
                  [255, 91, 0], [255, 122, 0], [255, 244, 0], [255, 0, 61], [255, 0, 152], [0, 10, 255], [255, 0, 91],
                  [234, 255, 0], [255, 153, 0], [0, 255, 10], [0, 193, 255], [0, 255, 224], [0, 102, 255],
                  [255, 183, 0], [81, 0, 255], [0, 255, 255], [51, 255, 0]]
    else:
        colors = [list((matplotlib.colors.hsv_to_rgb([x, 1.0, 1.0]) * 255).astype(int)) for x in
                  np.arange(0, 1, 1.0 / N)]
        np.random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def apply_parsing(image, part, color_map, alpha=0.7):
    """Apply the given parsing to the image.
    """
    assert image.shape[0] == part.shape[0] and image.shape[1] == part.shape[1]
    coordinates = np.where(part > 0)
    for i in range(len(coordinates[0])):
        image[coordinates[0][i], coordinates[1][i], :] = color_map[part[
            coordinates[0][i], coordinates[1][i]]] * alpha + image[coordinates[0][i], coordinates[1][i], :] * (
                                                                 1 - alpha)
    return image


def display_parsing(image, parsing, figsize=(16, 16), ax=None):
    """
    parsing: [height, width]
    figsize: (optional) the size of the image.
    """

    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)

    # Generate random colors
    colors = random_colors_opencv(20)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')

    masked_image = image.astype(np.uint32).copy()

    masked_image = apply_parsing(masked_image, parsing, colors)

    ax.imshow(masked_image.astype(np.uint8))


def display_instances(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2)] in image coordinates.
    masks: [height, width, num_instance]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    figsize: (optional) the size of the image.
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)

    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                              alpha=0.7, linestyle="dashed",
                              edgecolor=color, facecolor='none')
        ax.add_patch(p)

        # Label
        class_id = class_ids[i]
        score = scores[i] if scores is not None else None
        label = class_names[class_id]
        x = random.randint(x1, (x1 + x2) // 2)
        caption = "{} {:.3f}".format(label, score) if score else label
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))


def get_color_map(n=256):
    color_map = np.zeros((n, 3))
    for i in range(n):
        r = b = g = 0
        cid = i
        for j in range(0, 8):
            r = np.bitwise_or(r, np.left_shift(np.unpackbits(np.array([cid], dtype=np.uint8))[-1], 7 - j))
            g = np.bitwise_or(g, np.left_shift(np.unpackbits(np.array([cid], dtype=np.uint8))[-2], 7 - j))
            b = np.bitwise_or(b, np.left_shift(np.unpackbits(np.array([cid], dtype=np.uint8))[-3], 7 - j))
            cid = np.right_shift(cid, 3)

        color_map[i][0] = r
        color_map[i][1] = g
        color_map[i][2] = b
    return color_map


def write_part_result(res_dir, color_dir, height, width, image_id, global_parsing_prob, iscolor=True):
    """save the human parsing results (instance independent) in "vp_results/videoid/global_parsing/imageid.png"
    and save visualization results in "color_results/videoid/color/global_imageid.png"

    Input:
        global_parsing_prob: [height, width, NUM_PART_CLASS=20]

    Returns:
        global_parsing: shape [height, width], the value in [0-19],0 represent the background,
            1-19 represents the person part label
        global_parsing_max_prob: shape [height, width], the value in [0-1] represent the max probility of person part
            except background, 0 represent background
        global_parsing_map: shape [height, width, 3], which is used for visualized, 3 represent rgb,
    """
    if iscolor:
        if not os.path.exists(os.path.join(color_dir, 'color')):
            os.makedirs(os.path.join(color_dir, 'color'))
    if not os.path.exists(os.path.join(res_dir, 'global_parsing')):
        os.makedirs(os.path.join(res_dir, 'global_parsing'))
    c_map = np.array(random_colors_opencv(20))
    global_parsing = np.argmax(global_parsing_prob, axis=-1)  # shape [height, width],
    img_global_path = os.path.join(res_dir, "global_parsing", "%s.png" % image_id)
    if not os.path.exists(img_global_path):
        cv2.imwrite(img_global_path, global_parsing)
        # for getting global_parsing_max_prob and global_parsing_map
    global_parsing_max_prob = np.max(global_parsing_prob, axis=-1)  # shape [height, width]
    forground_map = (global_parsing > 0)
    global_parsing_max_prob = np.multiply(global_parsing_max_prob, forground_map)
    global_parsing_map = np.zeros((height, width, 3), dtype=np.uint8)
    if iscolor:
        coo = np.where(global_parsing > 0)  # tuple, len 2, coo[0] axis=0 index, coo[1] axis=1 index
        global_parsing_map[coo[0], coo[1], :] = c_map[global_parsing[coo[0], coo[1]]]
        color_global_path = os.path.join(color_dir, "color", "global_%s.png" % image_id)
        if not os.path.exists(color_global_path):
            cv2.imwrite(color_global_path, global_parsing_map)

    return global_parsing, global_parsing_max_prob, global_parsing_map


def write_inst_result(res_dir, color_dir, height, width, image_id, boxes, masks, scores, nms_like_thre=0.7):
    """
    (1) save the instance segmentation results in "vp_results/videoid/instance_segmentation/imageid.png"
    (2) save the detected person prob and bounding box(y1, x1, y2, x2) in vp_results/videoid/instance_segmentation/
        imageid.txt"
    (3) save visualization results in "color_results/videoid/color/inst_imageid.png"

    Args:
        boxes: [num_instance, (y1, x1, y2, x2)] in image coordinates.
        masks: [height, width, num_instance] of uint8
        scores: [num_instance] confidence scores for each box

    Returns:
        gray_map: ndarray, shape [height, width], the value is [0, num_instance], 0 is background,
            1-num_instance is the person label
        scores_boxes: [num_instance, (score, y1, x1, y2, x2)]
        color_map: [height, width, 3], the inst result will save in "color_results/videoid/color/inst_imageid.png"
    """
    if not os.path.exists(os.path.join(color_dir, 'color')):
        os.makedirs(os.path.join(color_dir, 'color'))
    if not os.path.exists(os.path.join(res_dir, 'instance_segmentation')):
        os.makedirs(os.path.join(res_dir, 'instance_segmentation'))
    t1 = time()
    c_map = get_color_map()
    t2 = time()
    # print(t2 - t1)
    masks = np.transpose(masks, (2, 0, 1))
    N = boxes.shape[0]
    color_map = np.zeros((height, width, 3), dtype=np.uint8)
    gray_map = np.zeros((height, width), dtype=np.uint8)
    inst_count = 1
    scores_boxes = []
    txt_instance_segmentation = os.path.join(res_dir, 'instance_segmentation', '%s.txt' % image_id)
    # if not os.path.exists(txt_instance_segmentation):
    # (2)
    wfp = open(txt_instance_segmentation, 'w')
    t3 = time()
    # print(t3 - t2)
    for i in range(N):
        mask = masks[i]
        box = boxes[i]
        score = scores[i]
        coo = np.where(mask > 0)
        u_pixels = len(coo[0])

        # nms-like postprocess, 假设有2个人，可能预测的2个mask中有部分地方有重叠，则处理第二个人时把重叠的部分设置为0
        for k in range(len(coo[0])):
            if gray_map[coo[0][k], coo[1][k]] > 0:
                mask[coo[0][k], coo[1][k]] = 0
        coo = np.where(mask > 0)
        u1_pixels = len(coo[0])

        if float(u1_pixels) / float(u_pixels) <= nms_like_thre:  # 由于重叠部分过多则直接舍弃到该实例mask
            continue

        # write score and bbox
        scores_boxes.append([score, box[0], box[1], box[2], box[3]])
        wfp.write('%f %d %d %d %d\n' % (score, box[0], box[1], box[2], box[3]))
        for k in range(len(coo[0])):
            gray_map[coo[0][k], coo[1][k]] = inst_count
            color_map[coo[0][k], coo[1][k], :] = c_map[inst_count][::-1]
        inst_count += 1
    # print("for", time() - t3)
    wfp.close()
    # (1)
    img_instance_segmentation_path = os.path.join(res_dir, "instance_segmentation", "%s.png" % image_id)
    if not os.path.exists(img_instance_segmentation_path):
        cv2.imwrite(img_instance_segmentation_path, gray_map)
    # (3)
    color_instance_segmentation_path = os.path.join(color_dir, "color", "inst_%s.png" % image_id)
    if not os.path.exists(color_instance_segmentation_path):
        cv2.imwrite(color_instance_segmentation_path, color_map)

    return gray_map, scores_boxes, color_map


def write_inst_result_v2(res_dir, color_dir, height, width, image_id, boxes, masks, scores, nms_like_thre=0.7):
    """
    (1) save the instance segmentation results in "vp_results/videoid/instance_segmentation/imageid.png"
    (2) save the dected person prob and bounding box(y1, x1, y2, x2) in vp_results/videoid/instance_segmentation/
        imageid.txt"
    (3) save visualization results in "color_results/videoid/color/inst_imageid.png"

    Args:
        boxes: [num_instance, (y1, x1, y2, x2)] in image coordinates.
        masks: [height, width, num_instance] of uint8
        scores: [num_instance] confidence scores for each box

    Returns:
        gray_map: ndarray, shape [height, width], the value is [0, num_instance], 0 is background,
            1-num_instance is the person label
        scores_boxes: [num_instance, (score, y1, x1, y2, x2)]
        color_map: [height, width, 3], the inst result will save in "color_results/videoid/color/inst_imageid.png"
    """
    if not os.path.exists(os.path.join(color_dir, 'color')):
        os.makedirs(os.path.join(color_dir, 'color'))
    if not os.path.exists(os.path.join(res_dir, 'instance_segmentation')):
        os.makedirs(os.path.join(res_dir, 'instance_segmentation'))
    t1 = time()
    c_map = get_color_map()
    t2 = time()
    # print(t2 - t1)
    masks = np.transpose(masks, (2, 0, 1))
    N = boxes.shape[0]
    color_map = np.zeros((height, width, 3), dtype=np.uint8)
    gray_map = np.zeros((height, width), dtype=np.uint8)
    inst_count = 1
    scores_boxes = []
    txt_instance_segmentation = os.path.join(res_dir, 'instance_segmentation', '%s.txt' % image_id)
    # if not os.path.exists(txt_instance_segmentation):
    # (2)
    wfp = open(txt_instance_segmentation, 'w')
    t3 = time()
    # print(t3 - t2)
    for i in range(N):
        mask = masks[i]
        box = boxes[i]
        score = scores[i]
        coo = np.where(mask > 0)
        u_pixels = len(coo[0])

        # nms-like postprocess
        for k in range(len(coo[0])):
            if gray_map[coo[0][k], coo[1][k]] > 0:
                mask[coo[0][k], coo[1][k]] = 0
        coo = np.where(mask > 0)
        u1_pixels = len(coo[0])

        if float(u1_pixels) / float(u_pixels) <= nms_like_thre:
            continue

        # write score and bbox
        scores_boxes.append([score, box[0], box[1], box[2], box[3]])
        wfp.write('%f %d %d %d %d\n' % (score, box[0], box[1], box[2], box[3]))
        for k in range(len(coo[0])):
            gray_map[coo[0][k], coo[1][k]] = inst_count
            color_map[coo[0][k], coo[1][k], :] = c_map[inst_count][::-1]
        inst_count += 1
    # print("for", time() - t3)
    wfp.close()
    # (1)
    img_instance_segmentation_path = os.path.join(res_dir, "instance_segmentation", "%s.png" % image_id)
    if not os.path.exists(img_instance_segmentation_path):
        cv2.imwrite(img_instance_segmentation_path, gray_map)
    # (3)
    color_instance_segmentation_path = os.path.join(color_dir, "color", "inst_%s.png" % image_id)
    if not os.path.exists(color_instance_segmentation_path):
        cv2.imwrite(color_instance_segmentation_path, color_map)

    return gray_map, scores_boxes, color_map


def write_inst_result_quickly(res_dir, color_dir, height, width, image_id, boxes, masks, scores, nms_like_thre=0.7,
                              iscolor=True):
    """
    (1) save the instance segmentation results in "vp_results/videoid/instance_segmentation/imageid.png"
    (2) save the detected person prob and bounding box(y1, x1, y2, x2) in vp_results/videoid/instance_segmentation/
        imageid.txt"
    (3) save visualization results in "color_results/videoid/color/inst_imageid.png"

    Args:
        boxes: [num_instance, (y1, x1, y2, x2)] in image coordinates.
        masks: [height, width, num_instance] of uint8
        scores: [num_instance] confidence scores for each box

    Returns:
        gray_map: ndarray, shape [height, width], the value is [0, num_instance], 0 is background,
            [1 - num_instance] is the person label
        scores_boxes: [num_instance, (score, y1, x1, y2, x2)]
        color_map: [height, width, 3], the inst result will save in "color_results/videoid/color/inst_imageid.png"
    """
    if iscolor:
        if not os.path.exists(os.path.join(color_dir, 'color')):
            os.makedirs(os.path.join(color_dir, 'color'))
    if not os.path.exists(os.path.join(res_dir, 'instance_segmentation')):
        os.makedirs(os.path.join(res_dir, 'instance_segmentation'))
    N = boxes.shape[0]
    t1 = time()
    c_map = random_colors_opencv(N)
    t2 = time()
    # print(t2 - t1)
    masks = np.transpose(masks, (2, 0, 1))
    color_map = np.zeros((height, width, 3), dtype=np.uint8)
    gray_map = np.zeros((height, width), dtype=np.uint8)
    inst_count = 1
    scores_boxes = []
    txt_instance_segmentation = os.path.join(res_dir, 'instance_segmentation', '%s.txt' % image_id)
    # if not os.path.exists(txt_instance_segmentation):
    wfp = open(txt_instance_segmentation, 'w')
    t3 = time()
    # print(t3 - t2)
    for i in range(N):
        mask = masks[i]
        box = boxes[i]
        score = scores[i]
        coo = np.where(mask > 0)
        u_pixels = len(coo[0])

        # nms-like postprocess
        mask[(mask > 0) & (gray_map > 0)] = 0
        coo = np.where(mask > 0)
        u1_pixels = len(coo[0])

        if float(u1_pixels) / float(u_pixels) <= nms_like_thre:
            continue

        # write score and bbox
        scores_boxes.append([score, box[0], box[1], box[2], box[3]])
        wfp.write('%f %d %d %d %d\n' % (score, box[0], box[1], box[2], box[3]))
        gray_map[mask > 0] = inst_count
        if iscolor:
            color_map[coo[0], coo[1], :] = c_map[inst_count]
        inst_count += 1
    # print("for", time() - t3)
    wfp.close()
    img_instance_segmentation_path = os.path.join(res_dir, "instance_segmentation", "%s.png" % image_id)
    if not os.path.exists(img_instance_segmentation_path):
        cv2.imwrite(img_instance_segmentation_path, gray_map)
    if iscolor:
        color_instance_segmentation_path = os.path.join(color_dir, "color", "inst_%s.png" % image_id)
        if not os.path.exists(color_instance_segmentation_path):
            cv2.imwrite(color_instance_segmentation_path, color_map)

    return gray_map, scores_boxes, color_map


def write_inst_part_result(res_dir, color_dir, height, width, image_id, boxes, masks, scores, global_parsing_prob,
                           nms_like_thre=0.7, class_num=20, is_combine_inst_part=True, iscolor=True):
    """
        A. write_part_result function:
            (1)the human global parsing results (instance independent) in "vp_results/videoid/global_parsing/imageid.png"
            (2)save visualization results in "color_results/videoid/color/global_imageid.png"
        B. write_inst_result_quickly or write_inst_result function:
            (1) save the instance segmentation results in "vp_results/videoid/instance_segmentation/imageid.png"
            (2) save the detected person prob and bounding box(y1, x1, y2, x2) in vp_results/videoid/instance_segmentation/
                imageid.txt"
            (3) save visualization results in "color_results/videoid/color/inst_imageid.png"
        C. is_combine_inst_part is True
            (1) save the combined inst and part results into "vp_results/videoid/instance_parsing/imageid.png"
            (2) save the every person and every part results into "vp_results/videoid/instance_parsing/imageid.txt"

    Args:
        res_dir: str, save into os.path.join(RES_DIR, "vp_results", videoid) such as RES_DIR/vp_results/videos45
        color_dir: str, save into os.path.join(RES_DIR, "color_results", videoid) such as RES_DIR/color_results/videos45
        height: int, for VIP is 720
        width: int, for VIP is 1280
       image_id: str, for example '000000000176'
        boxes: numpy.ndarray, shape=[num_instance, (y1, x1, y2, x2)] in image coordinates.
        masks: numpy.ndarray, shape=[height, width, num_instance] (720, 1280, 3),dtype=uint8
        scores: [num_instance] confidence scores for each box
        global_parsing_prob: [height=720, width=1280, NUM_PART_CLASS=20]
        nms_like_thre: float, default 0.7, used in write_inst_result_quickly function or write_inst_result function
        class_num: int, for VIP is 20=1+19, 1 is bg, 19 is person part label num
        is_combine_inst_part: bool, default True, for test, need save the instance parsing result

    Returns:
        global_parsing_map, color_map
    """
    t0 = time()
    # global_parsing: shape [height, width], the value in [0-19],0 represent the background,
    #   1-19 represents the person part label
    # global_parsing_max_prob: shape [height, width], the value in [0-1] represent the max probility of person part
    #   except background, 0 represent background
    # global_parsing_map: shape [height, width, 3], which is used for visualized, 3 represent rgb,
    global_parsing, global_parsing_max_prob, global_parsing_map = write_part_result(res_dir, color_dir, height, width,
                                                                                    image_id, global_parsing_prob,
                                                                                    iscolor=iscolor)
    t1 = time()
    print("        A. write_part_result:", t1 - t0)
    # inst_map, inst_scores, color_map = write_inst_result(res_dir, color_dir, height, width, image_id, boxes, masks,
    #                                                      scores, nms_like_thre)

    # inst_map(gray_map): ndarray, shape [height, width], the value is [0, num_instance], 0 is background,
    #   [1 - num_instance] is the person label
    # inst_scores(scores_boxes): [num_instance, (score, y1, x1, y2, x2)]
    # color_map: [height, width, 3], the inst result will save in "color_results/videoid/color/inst_imageid.png"
    inst_map, inst_scores, color_map = write_inst_result_quickly(res_dir, color_dir, height, width, image_id, boxes,
                                                                 masks, scores, nms_like_thre, iscolor=iscolor)
    t2 = time()
    print("        B. write_inst_result:", t2 - t1)
    # C. combine inst and part, need `global_parsing`, `global_parsing_max_prob`, `inst_map`, `inst_scores`

    # inst_part_map:shape [height, width], the value is counter used label, every person every part have different label
    inst_part_map = np.zeros_like(inst_map)  # shape [height, width], will save the result into picture
    if not is_combine_inst_part:
        return global_parsing_map, color_map
    floder = os.path.join(res_dir, 'instance_parsing')
    if not os.path.exists(floder):
        os.makedirs(floder)
    instance_parsing_path = '%s/%s.txt' % (floder, image_id)
    wfp = open(instance_parsing_path, 'w')
    counter = 0
    t3 = time()
    for k in range(1, class_num):  # class_num=20
        cur_counter = counter
        inst_part_prob_map = {}  # dict, key is counter(every person every part have different label), value is float
        cls_indices = (global_parsing == k).astype(np.uint8)  # shape [height, width]
        part_inst_map = cls_indices * inst_map  # the person inst with the same part label
        inst_ids = np.unique(part_inst_map)  # for example [0 2 3], 0 is bg, 2 inst person label, 3 is the same to 2
        tt0 = time()
        for i in inst_ids:
            if i != 0:
                tt1 = time()
                counter = counter + 1
                cls_inst_indices = np.where(part_inst_map == i)  # tuple, len is 2
                inst_part_map[cls_inst_indices] = counter

                human_id = i
                human_seg_sco = inst_scores[human_id - 1][0]

                tmp_parsing_prob = global_parsing_max_prob[cls_inst_indices]
                mean_parsing_prob = np.mean(tmp_parsing_prob)

                inst_part_prob_map[counter] = mean_parsing_prob * human_seg_sco
                # print("tt1", time() - tt1, "s")
        tt2 = time()
        # print("tt", tt2 - tt0, "s")
        if cur_counter < counter:
            for i in range(cur_counter, counter):
                wfp.write('%d %f\n' % (k, inst_part_prob_map[i + 1]))  # k is part label
        # print("write", time() - tt2, "s")
    wfp.close()
    img_instance_parsing_path = os.path.join(floder, "%s.png" % image_id)
    if not os.path.exists(img_instance_parsing_path):
        cv2.imwrite(img_instance_parsing_path, inst_part_map)
    t4 = time()
    print("        C. combine every person and every part:", t4 - t3, "s")
    return global_parsing_map, color_map


def vis_inst_parsings(image, res_dir, image_id, boxes, parts, class_ids,
                      scores=None, class_names=['BG', 'person'],
                      figsize=(16, 16)):
    """
    boxes: [num_instance, (y1, x1, y2, x2)] in image coordinates.
    parts: [height, width, num_instance]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    figsize: (optional) the size of the image.
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == parts.shape[-1] == class_ids.shape[0]

    fig, ax = plt.subplots(1, figsize=figsize)

    # Generate random colors
    color_map = get_color_map()
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    # ax.set_ylim(height + 10, -10)
    # ax.set_xlim(-10, width + 10)
    # ax.axis('off')
    # ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                              alpha=0.7, linestyle="dashed",
                              edgecolor=color, facecolor='none')
        ax.add_patch(p)

        # Label
        class_id = class_ids[i]
        score = scores[i] if scores is not None else None
        label = class_names[class_id]
        x = random.randint(x1, (x1 + x2) // 2)
        caption = "{} {:.3f}".format(label, score) if score else label
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

        # Mask
        part = parts[:, :, i]
        masked_image = apply_parsing(masked_image, part, color_map)

    ax.imshow(masked_image.astype(np.uint8))
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    fig.savefig(os.path.join(res_dir, 'vis_%s.png' % image_id))
    plt.close()


def vis_insts(image, res_dir, image_id, boxes, masks, class_ids,
              scores=None, class_names=['BG', 'person'], figsize=(16, 16)):
    """discarded
    Args:
        image: numpy.ndarray, shape=[height, width, num_instance] (720, 1280, 3), dtype=uint32
        boxes: numpy.ndarray, shape=[num_instance, (y1, x1, y2, x2)] in image coordinates.
        masks: numpy.ndarray, shape=[height, width, num_instance] (720, 1280, 3),dtype=uint8
        class_ids: list, len=num_instances
        class_names: list of class names of the dataset, default is ['BG', 'person']
        scores: (optional) confidence scores for each box,numpy.ndarray,shape=[num_instances], dtype=float32
        figsize: (optional) the size of the image.
    """
    # Number of 
    if not os.path.exists(os.path.join(res_dir, 'color')):
        os.makedirs(os.path.join(res_dir, 'color'))

    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    fig, ax = plt.subplots(1, figsize=figsize)

    # Generate random colors
    colors = random_colors(N)
    # colors = random_colors_opencv(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]

    masked_image = image.astype(np.uint32).copy()
    t1 = time()
    print("N", N)
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                              alpha=0.7, linestyle="dashed",
                              edgecolor=color, facecolor='none')
        ax.add_patch(p)

        # Label
        class_id = class_ids[i]
        score = scores[i] if scores is not None else None
        label = class_names[class_id]
        x = random.randint(x1, (x1 + x2) // 2)
        caption = "{} {:.3f}".format(label, score) if score else label
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    t2 = time()
    print(t2 - t1)
    ax.imshow(masked_image.astype(np.uint8))
    t3 = time()
    print(t3 - t2)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    img_path = os.path.join(res_dir, 'color', 'vis_%s.png' % image_id)
    if not os.path.exists(img_path):
        fig.savefig(img_path)
    plt.close()
    t4 = time()
    print("savefig", t4 - t3)
    return masked_image


def vis_insts_opencv(image, color_dir, image_id, boxes, masks, class_ids,
                     scores=None, class_names=['BG', 'person'], figsize=(16, 16)):
    """write a bounding box for every person and contour, the image saves in RES_DIR/color_results/videoid/vis_%s.png

   Args:
       image: numpy.ndarray, shape=[height, width, num_instance] (720, 1280, 3), dtype=uint8,3 represent rgb,
       color_dir: str, save the os.path.join(RES_DIR, "color_results", videoid) such as RES_DIR/color_results/videos45
       image_id: str, for example '000000000176'
       boxes: numpy.ndarray, shape=[num_instance, (y1, x1, y2, x2)] in image coordinates.
       masks: numpy.ndarray, shape=[height, width, num_instance] (720, 1280, 3),dtype=uint8
       class_ids: list, len=num_instances
       class_names: list of class names of the dataset, default is ['BG', 'person']
       scores: (optional) confidence scores for each box,numpy.ndarray,shape=[num_instances], dtype=float32
       figsize: (not used) the size of the image.
    Returns:
        masked_image: shape [height, widht, 3], 3 represent rgb, wthe image including the bounding box
    """
    # create the saved dir
    if not os.path.exists(os.path.join(color_dir, 'color')):
        os.makedirs(os.path.join(color_dir, 'color'))
    # Number of person
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # Generate random colors
    colors = random_colors_opencv(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]

    masked_image = image.astype(np.uint8).copy()
    t1 = time()
    # print("N", N)
    for i in range(N):
        color = colors[i]
        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        cv2.rectangle(masked_image, (x1, y1), (x2, y2), color=color, thickness=1)

        # Label
        class_id = class_ids[i]
        score = scores[i] if scores is not None else None
        label = class_names[class_id]
        x = random.randint(x1, (x1 + x2) // 2)
        caption = "{} {:.3f}".format(label, score) if score else label
        cv2.putText(masked_image, caption, (x1, max(y1 - 15, 0)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                    color=color, thickness=1)
        # Mask
        mask = masks[:, :, i]
        masked_image_binary = mask * 255
        masked_image_binary = masked_image_binary.astype(np.uint8)

        # Mask Polygon
        # opencv 找边缘轮廓需要传入二值图
        res = cv2.findContours(masked_image_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = res[-2]
        cv2.drawContours(masked_image, contours, -1, color=color, thickness=1)
    img_path = os.path.join(color_dir, 'color', 'vis_%s.png' % image_id)
    if not os.path.exists(img_path):
        cv2.imwrite(img_path, masked_image)
    t2 = time()
    # print("opencv", t2 - t1, "s")
    return masked_image


def draw_rois(image, rois, refined_rois, mask, class_ids, class_names, limit=10):
    """
    anchors: [n, (y1, x1, y2, x2)] list of anchors in image coordinates.
    proposals: [n, 4] the same anchors but refined to fit objects better.
    """
    masked_image = image.copy()

    # Pick random anchors in case there are too many.
    ids = np.arange(rois.shape[0], dtype=np.int32)
    ids = np.random.choice(
        ids, limit, replace=False) if ids.shape[0] > limit else ids

    fig, ax = plt.subplots(1, figsize=(12, 12))
    if rois.shape[0] > limit:
        plt.title("Showing {} random ROIs out of {}".format(
            len(ids), rois.shape[0]))
    else:
        plt.title("{} ROIs".format(len(ids)))

    # Show area outside image boundaries.
    ax.set_ylim(image.shape[0] + 20, -20)
    ax.set_xlim(-50, image.shape[1] + 20)
    ax.axis('off')

    for i, id in enumerate(ids):
        color = np.random.rand(3)
        class_id = class_ids[id]
        # ROI
        y1, x1, y2, x2 = rois[id]
        p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                              edgecolor=color if class_id else "gray",
                              facecolor='none', linestyle="dashed")
        ax.add_patch(p)
        # Refined ROI
        if class_id:
            ry1, rx1, ry2, rx2 = refined_rois[id]
            p = patches.Rectangle((rx1, ry1), rx2 - rx1, ry2 - ry1, linewidth=2,
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)
            # Connect the top-left corners of the anchor and proposal for easy visualization
            ax.add_line(lines.Line2D([x1, rx1], [y1, ry1], color=color))

            # Label
            label = class_names[class_id]
            ax.text(rx1, ry1 + 8, "{}".format(label),
                    color='w', size=11, backgroundcolor="none")

            # Mask
            m = utils.unmold_mask(mask[id], rois[id]
            [:4].astype(np.int32), image.shape)
            masked_image = apply_mask(masked_image, m, color)

    ax.imshow(masked_image)

    # Print stats
    print("Positive ROIs: ", class_ids[class_ids > 0].shape[0])
    print("Negative ROIs: ", class_ids[class_ids == 0].shape[0])
    print("Positive Ratio: {:.2f}".format(
        class_ids[class_ids > 0].shape[0] / class_ids.shape[0]))


# TODO: Replace with matplotlib equivalent?
def draw_box(image, box, color):
    """Draw 3-pixel width bounding boxes on the given image array.
    color: list of 3 int values for RGB.
    """
    y1, x1, y2, x2 = box
    image[y1:y1 + 2, x1:x2] = color
    image[y2:y2 + 2, x1:x2] = color
    image[y1:y2, x1:x1 + 2] = color
    image[y1:y2, x2:x2 + 2] = color
    return image


def display_top_masks(image, mask, class_ids, class_names, limit=4):
    """Display the given image and the top few class masks."""
    to_display = []
    titles = []
    to_display.append(image)
    titles.append("H x W={}x{}".format(image.shape[0], image.shape[1]))
    # Pick top prominent classes in this image
    unique_class_ids = np.unique(class_ids)
    mask_area = [np.sum(mask[:, :, np.where(class_ids == i)[0]])
                 for i in unique_class_ids]
    top_ids = [v[0] for v in sorted(zip(unique_class_ids, mask_area),
                                    key=lambda r: r[1], reverse=True) if v[1] > 0]
    # Generate images and titles
    for i in range(limit):
        class_id = top_ids[i] if i < len(top_ids) else -1
        # Pull masks of instances belonging to the same class.
        m = mask[:, :, np.where(class_ids == class_id)[0]]
        m = np.sum(m * np.arange(1, m.shape[-1] + 1), -1)
        to_display.append(m)
        titles.append(class_names[class_id] if class_id != -1 else "-")
    display_images(to_display, titles=titles, cols=limit + 1, cmap="Blues_r")


def plot_precision_recall(AP, precisions, recalls):
    """Draw the precision-recall curve.

    AP: Average precision at IoU >= 0.5
    precisions: list of precision values
    recalls: list of recall values
    """
    # Plot the Precision-Recall curve
    _, ax = plt.subplots(1)
    ax.set_title("Precision-Recall Curve. AP@50 = {:.3f}".format(AP))
    ax.set_ylim(0, 1.1)
    ax.set_xlim(0, 1.1)
    _ = ax.plot(recalls, precisions)


def plot_overlaps(gt_class_ids, pred_class_ids, pred_scores,
                  overlaps, class_names, threshold=0.5):
    """Draw a grid showing how ground truth objects are classified.
    gt_class_ids: [N] int. Ground truth class IDs
    pred_class_id: [N] int. Predicted class IDs
    pred_scores: [N] float. The probability scores of predicted classes
    overlaps: [pred_boxes, gt_boxes] IoU overlaps of predictins and GT boxes.
    class_names: list of all class names in the dataset
    threshold: Float. The prediction probability required to predict a class
    """
    gt_class_ids = gt_class_ids[gt_class_ids != 0]
    pred_class_ids = pred_class_ids[pred_class_ids != 0]

    plt.figure(figsize=(12, 10))
    plt.imshow(overlaps, interpolation='nearest', cmap=plt.cm.Blues)
    plt.yticks(np.arange(len(pred_class_ids)),
               ["{} ({:.2f})".format(class_names[int(id)], pred_scores[i])
                for i, id in enumerate(pred_class_ids)])
    plt.xticks(np.arange(len(gt_class_ids)),
               [class_names[int(id)] for id in gt_class_ids], rotation=90)

    thresh = overlaps.max() / 2.
    for i, j in itertools.product(range(overlaps.shape[0]),
                                  range(overlaps.shape[1])):
        text = ""
        if overlaps[i, j] > threshold:
            text = "match" if gt_class_ids[j] == pred_class_ids[i] else "wrong"
        color = ("white" if overlaps[i, j] > thresh
                 else "black" if overlaps[i, j] > 0
        else "grey")
        plt.text(j, i, "{:.3f}\n{}".format(overlaps[i, j], text),
                 horizontalalignment="center", verticalalignment="center",
                 fontsize=9, color=color)

    plt.tight_layout()
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")


def draw_boxes(image, boxes=None, refined_boxes=None,
               masks=None, captions=None, visibilities=None,
               title="", ax=None):
    """Draw bounding boxes and segmentation masks with differnt
    customizations.

    boxes: [N, (y1, x1, y2, x2, class_id)] in image coordinates.
    refined_boxes: Like boxes, but draw with solid lines to show
        that they're the result of refining 'boxes'.
    masks: [N, height, width]
    captions: List of N titles to display on each box
    visibilities: (optional) List of values of 0, 1, or 2. Determine how
        prominant each bounding box should be.
    title: An optional title to show over the image
    ax: (optional) Matplotlib axis to draw on.
    """
    # Number of boxes
    assert boxes is not None or refined_boxes is not None
    N = boxes.shape[0] if boxes is not None else refined_boxes.shape[0]

    # Matplotlib Axis
    if not ax:
        _, ax = plt.subplots(1, figsize=(12, 12))

    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    margin = image.shape[0] // 10
    ax.set_ylim(image.shape[0] + margin, -margin)
    ax.set_xlim(-margin, image.shape[1] + margin)
    ax.axis('off')

    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        # Box visibility
        visibility = visibilities[i] if visibilities is not None else 1
        if visibility == 0:
            color = "gray"
            style = "dotted"
            alpha = 0.5
        elif visibility == 1:
            color = colors[i]
            style = "dotted"
            alpha = 1
        elif visibility == 2:
            color = colors[i]
            style = "solid"
            alpha = 1

        # Boxes
        if boxes is not None:
            if not np.any(boxes[i]):
                # Skip this instance. Has no bbox. Likely lost in cropping.
                continue
            y1, x1, y2, x2 = boxes[i]
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  alpha=alpha, linestyle=style,
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Refined boxes
        if refined_boxes is not None and visibility > 0:
            ry1, rx1, ry2, rx2 = refined_boxes[i].astype(np.int32)
            p = patches.Rectangle((rx1, ry1), rx2 - rx1, ry2 - ry1, linewidth=2,
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)
            # Connect the top-left corners of the anchor and proposal
            if boxes is not None:
                ax.add_line(lines.Line2D([x1, rx1], [y1, ry1], color=color))

        # Captions
        if captions is not None:
            caption = captions[i]
            # If there are refined boxes, display captions on them
            if refined_boxes is not None:
                y1, x1, y2, x2 = ry1, rx1, ry2, rx2
            x = random.randint(x1, (x1 + x2) // 2)
            ax.text(x1, y1, caption, size=11, verticalalignment='top',
                    color='w', backgroundcolor="none",
                    bbox={'facecolor': color, 'alpha': 0.5,
                          'pad': 2, 'edgecolor': 'none'})

        # Masks
        if masks is not None:
            mask = masks[:, :, i]
            masked_image = apply_mask(masked_image, mask, color)
            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))


def display_table(table):
    """Display values in a table format.
    table: an iterable of rows, and each row is an iterable of values.
    """
    html = ""
    for row in table:
        row_html = ""
        for col in row:
            row_html += "<td>{:40}</td>".format(str(col))
        html += "<tr>" + row_html + "</tr>"
    html = "<table>" + html + "</table>"
    IPython.display.display(IPython.display.HTML(html))


def display_weight_stats(model):
    """Scans all the weights in the model and returns a list of tuples
    that contain stats about each weight.
    """
    layers = model.get_trainable_layers()
    table = [["WEIGHT NAME", "SHAPE", "MIN", "MAX", "STD"]]
    for l in layers:
        weight_values = l.get_weights()  # list of Numpy arrays
        weight_tensors = l.weights  # list of TF tensors
        for i, w in enumerate(weight_values):
            weight_name = weight_tensors[i].name
            # Detect problematic layers. Exclude biases of conv layers.
            alert = ""
            if w.min() == w.max() and not (l.__class__.__name__ == "Conv2D" and i == 1):
                alert += "<span style='color:red'>*** dead?</span>"
            if np.abs(w.min()) > 1000 or np.abs(w.max()) > 1000:
                alert += "<span style='color:red'>*** Overflow?</span>"
            # Add row
            table.append([
                weight_name + alert,
                str(w.shape),
                "{:+9.4f}".format(w.min()),
                "{:+10.4f}".format(w.max()),
                "{:+9.4f}".format(w.std()),
            ])
    display_table(table)
