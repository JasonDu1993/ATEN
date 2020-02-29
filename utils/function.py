# -*- coding: utf-8 -*-
# @Time    : 2020/2/28 23:52
# @Author  : Jason
# @Email   : 1358681631@qq.com
# @File    : function.py
# @Software: PyCharm

import cv2
import numpy as np


def add_decoration(image, part_inst_map):
    """

    Args:
        image: bgr
        part_inst_map:

    Returns:
        image:

    """
    masked_image = image.astype(np.uint8).copy()
    h, w, _ = masked_image.shape
    part_inst_map = np.array(part_inst_map, dtype=np.uint8)
    ret, binary = cv2.threshold(part_inst_map, 0, 255, cv2.THRESH_BINARY)
    crown = cv2.imread("./utils/pic/crown.png")
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    crown_resizes = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    for i in range(0, len(contours)):
        # print(contours[i])
        startx, starty, width, height = cv2.boundingRect(contours[i])
        print(startx, starty, width, height)
        height = starty - max(0, starty - width)
        # crown_width, crown_height = width * 2 // 3, height * 2 // 3
        crown_width, crown_height = int(width * 0.9), int(height * 0.8)
        crown_centerx = startx + width // 2
        crown_centery = starty - height // 2
        crown_x1 = crown_centerx - crown_width // 2
        crown_y1 = crown_centery - crown_height // 2
        crown_x2 = crown_x1 + crown_width
        crown_y2 = crown_y1 + crown_height
        crown_resize = cv2.resize(crown, (crown_width, crown_height), interpolation=cv2.INTER_AREA)
        crown_mask = np.array(crown_resize == 0, dtype=np.uint8)
        # cv2.rectangle(masked_image, (startx, starty), (startx + width, starty + height), (153, 153, 0), 1)
        masked_image[crown_y1:crown_y2, crown_x1:crown_x2, :] = \
            cv2.copyTo(masked_image[crown_y1:crown_y2, crown_x1:crown_x2, :], crown_mask)
        masked_image[crown_y1:crown_y2, crown_x1:crown_x2, :] = masked_image[crown_y1:crown_y2, crown_x1:crown_x2, :] + \
                                                                crown_resize
    # masked_image = cv2.copyTo(masked_image, crown_resizes)
    # cv2.imshow("img", masked_image)
    # cv2.waitKey(0)
    return masked_image
