import os
import numpy as np
from PIL import Image
import time

n_cl = 20
CLASSES = ['background', 'hat', 'hair', 'sun-glasses', 'upper-clothes', 'dress',
           'coat', 'socks', 'pants', 'gloves', 'scarf', 'skirt', 'torso-skin',
           'face', 'right-arm', 'left-arm', 'right-leg', 'left-leg', 'right-shoe', 'left-shoe']
# GT_DIR = '/your/path/to/VIP/Category_ids'
# PRE_DIR = '/your/path/to/results'
"""
Category_ids:存储人体解析身体部位的注解part_anno，其中每个部位一个数字表示，
此次不区分不同的人，即每个人的部位数字是一样的
"""

PRE_DIR = "/home/sk49/workspace/zhoudu/ATEN/vis_mfp/val_mfp_20191210b_epoch041/vp_results"
NAME = "val_mfp_20191210b_epoch041"
TMP_DIR = "./eval_results"

GT_DIR = '/home/sk49/workspace/dataset/VIP/Category_ids'
evalute_result_path = os.path.join(TMP_DIR, NAME, "eval_" + NAME + "_parsing.txt")
if not os.path.exists(os.path.join(TMP_DIR, NAME)):
    os.makedirs(os.path.join(TMP_DIR, NAME))
f = open(evalute_result_path, "w")
res = ""


def main():
    image_paths, label_paths = init_path()
    hist = compute_hist(image_paths, label_paths)
    show_result(hist)


def _get_voc_color_map(n=256):
    color_map = np.zeros((n, 3))
    index_map = {}
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
        index_map['%d_%d_%d' % (r, g, b)] = i
    return color_map, index_map


def init_path():
    image_dir = PRE_DIR
    label_dir = GT_DIR

    file_names = []
    for vid in os.listdir(image_dir):
        if not vid.startswith("video"):
            continue
        for img in os.listdir(os.path.join(image_dir, vid, 'global_parsing')):
            # if img.startswith("video"):
            file_names.append([vid, img[:-4]])
    global res
    res += "video name:" + str(len(file_names)) + "\n"
    res += "result of" + image_dir + "\n"
    # print("video name:", len(file_names), file_names)
    print("result of", image_dir)

    image_paths = []
    label_paths = []
    for file_name in file_names:
        image_paths.append(os.path.join(image_dir, file_name[0], 'global_parsing', file_name[1] + '.png'))
        label_paths.append(os.path.join(label_dir, file_name[0], file_name[1] + '.png'))
    return image_paths, label_paths


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def compute_hist(images, labels):
    color_map, index_map = _get_voc_color_map()
    hist = np.zeros((n_cl, n_cl))
    for img_path, label_path in zip(images, labels):
        label = Image.open(label_path)
        label_array = np.array(label, dtype=np.int32)
        image = Image.open(img_path)
        image_array = np.array(image, dtype=np.int32)

        gtsz = label_array.shape

        imgsz = image_array.shape
        if not gtsz == imgsz:
            image = image.resize((gtsz[1], gtsz[0]), Image.ANTIALIAS)
            image_array = np.array(image, dtype=np.int32)

        hist += fast_hist(label_array, image_array, n_cl)

    return hist


def show_result(hist):
    classes = CLASSES
    # num of correct pixels
    num_cor_pix = np.diag(hist)
    # num of gt pixels
    num_gt_pix = hist.sum(1)

    global res
    print('=' * 50)

    # @evaluation 1: overall accuracy
    acc = num_cor_pix.sum() / hist.sum()
    print('>>>', 'overall accuracy', acc)
    res += "1, overall accuracy:\n"
    res += ">>> acc: " + str(acc) + "\n"
    res += '-' * 50 + "\n"
    print('-' * 50)

    # @evaluation 2: mean accuracy & per-class accuracy
    res += "2, Accuracy for each class (pixel accuracy), mean accuracy & per-class accuracy:\n"
    print('Accuracy for each class (pixel accuracy):')
    for i in range(n_cl):
        print('%-15s: %f' % (classes[i], num_cor_pix[i] / num_gt_pix[i]))
        res += '%-15s: %f' % (classes[i], num_cor_pix[i] / num_gt_pix[i]) + "\n"
    acc = num_cor_pix / num_gt_pix
    print('>>>', 'mean accuracy', np.nanmean(acc))
    res += ">>> mean accuracy: " + str(np.nanmean(acc)) + "\n"
    res += '-' * 50 + "\n"
    print('-' * 50)

    # @evaluation 3: mean IU & per-class IU
    res += "3, mean IU & per-class IU:\n"
    union = num_gt_pix + hist.sum(0) - num_cor_pix
    for i in range(n_cl):
        print('%-15s: %f' % (classes[i], num_cor_pix[i] / union[i]))
        res += '%-15s: %f' % (classes[i], num_cor_pix[i] / union[i]) + "\n"
    iu = num_cor_pix / (num_gt_pix + hist.sum(0) - num_cor_pix)
    res += ">>> mean IU: " + str(np.nanmean(iu)) + "\n"
    res += '-' * 50 + "\n"
    print('>>>', 'mean IU', str(np.nanmean(iu)))
    print('-' * 50)

    # @evaluation 4: frequency weighted IU
    res += "4, frequency weighted IU:\n"
    freq = num_gt_pix / hist.sum()
    res += ">>> IU: " + str((freq[freq > 0] * iu[freq > 0]).sum()) + "\n"
    res += '-' * 50 + "\n"
    print('>>>', 'IU', (freq[freq > 0] * iu[freq > 0]).sum())
    print('=' * 50)
    f.write(res)
    f.flush()
    f.close()


if __name__ == '__main__':
    # testing vip val dataset 2445 images spend almost 1 min
    t0 = time.time()
    main()
    print("total time", time.time() - t0, "s")
