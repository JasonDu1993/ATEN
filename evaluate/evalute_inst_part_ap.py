import time
import os
from PIL import Image
import numpy as np
import multiprocessing

PREDICT_DIR = "/home/sk49/workspace/zhoudu/ATEN/vis_hpa/val_hpa_20191201a_epoch033/vp_results"
NAME = "val_hpa_20191201a_epoch033"  # tmp class file
TMP_DIR = "./eval_results"
NUM_PROCESS = 10

# PREDICT_DIR = r'D:\workspaces\ATEN\vis\viptiny_test_eval\vp_results'
# INST_PART_GT_DIR = r'D:\dataset\VIP_tiny\Instance_ids'

INST_PART_GT_DIR = '/home/sk49/workspace/dataset/VIP/Instance_ids'
CLASSES = ['background', 'hat', 'hair', 'gloves', 'sun-glasses', 'upper-clothes', 'dress',
           'coat', 'socks', 'pants', 'torso-skin', 'scarf', 'skirt',
           'face', 'left-arm', 'right-arm', 'left-leg', 'right-leg', 'left-shoe', 'right-shoe']

IOU_THRE = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
if not os.path.exists(os.path.join(TMP_DIR, NAME)):
    os.makedirs(os.path.join(TMP_DIR, NAME))


# compute mask overlap
def compute_mask_iou(mask_gt, masks_pre, mask_gt_area, masks_pre_area):
    """Calculates IoU of the given box with the array of the given boxes.
    mask_gt: ndarray, [H,W] #一个gt的mask
    masks_pre: ndarray, [num_instances， height, width] predict Instance masks
    mask_gt_area: int, the gt_mask_area
    masks_pre_area: tuple, array of length masks_count.包含了所有预测的mask的sum

    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.
    """
    intersection = np.logical_and(mask_gt, masks_pre)
    intersection = np.where(intersection == True, 1, 0).astype(np.uint8)
    intersection = NonZero(intersection)

    # print('intersection  ：', intersection)

    mask_gt_areas = np.full(len(masks_pre_area), mask_gt_area)

    union = mask_gt_areas + masks_pre_area[:] - intersection[:]

    iou = intersection / (union + 1e-7)

    return iou


# 计算mask中所有非零个数
def NonZero(masks):
    """
    :param masks: [N,h,w]一个三维数组，里面放的是二维的mask数组
    :return: (N) 返回一个长度为N的tuple(元素不能修改)，里面是对应的二维mask 中非零位置数目
    """
    area = []  # 先定义成list，返回时修改
    # print('NonZero masks',masks.shape)
    for i in masks:
        _, a = np.nonzero(i)
        area.append(a.shape[0])
    area = tuple(area)
    return area


def compute_mask_overlaps(masks_pre, masks_gt):
    """Computes IoU overlaps between two sets of boxes.
    For better performance, pass the largest set first and the smaller second.
    Args:
        masks_pre, masks_gt:
        masks_pre 表示待计算的mask [num_instances_pre,height, width] Instance masks
        masks_gt 表示的是ground truth [num_instances_gt,height, width]
    Returns:
        overlaps: ndarray, shape [num_instances_pre, num_instances_gt]
    """
    # Areas of masks_pre and masks_gt 获得所有非零数值个数
    area1 = NonZero(masks_pre)
    area2 = NonZero(masks_gt)

    # print(masks_pre.shape, masks_gt.shape)(1, 375, 1) (500, 375, 1)

    # Compute overlaps to generate matrix [masks count, masks_gt count]
    # Each cell contains the IoU value.
    # print('area1',len(area1))
    # print('area1',len(area2))
    overlaps = np.zeros((masks_pre.shape[0], masks_gt.shape[0]))  # shape [num_instances_pre, num_instances_gt]
    # print('overlaps ： ',overlaps.shape)
    for i in range(overlaps.shape[1]):
        mask_gt = masks_gt[i]
        # print('overlaps：',overlaps)
        overlaps[:, i] = compute_mask_iou(mask_gt, masks_pre, area2[i], area1)

    return overlaps


def voc_ap(rec, prec, use_07_metric=False):
    """
    Compute VOC AP given precision and recall. If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    Args:
        rec: recall
        prec: precision
        use_07_metric:
    Returns:
        ap: average precision
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        # arange([start, ]stop, [step, ]dtype=None)
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap += p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def convert2evalformat(inst_id_map_gt, id_to_convert=None):
    """
    param:
        inst_id_map_gt:[h, w]
        id_to_convert: a set
    return:
        masks:[instances,h, w]
    """
    masks = []

    inst_ids = np.unique(inst_id_map_gt)
    # print("inst_ids:", inst_ids)
    background_ind = np.where(inst_ids == 0)[0]
    inst_ids = np.delete(inst_ids, background_ind)

    if id_to_convert is None:
        for i in inst_ids:
            im_mask = (inst_id_map_gt == i).astype(np.uint8)
            masks.append(im_mask)
    else:
        for i in inst_ids:
            if i not in id_to_convert:
                continue
            im_mask = (inst_id_map_gt == i).astype(np.uint8)
            masks.append(im_mask)

    return masks, len(masks)


def compute_class_ap(image_id_list, class_id, iou_threshold, save_path):
    """Compute Average Precision at a set IoU threshold (default 0.5).
    Args:
        image_id_list : list(list), all pictures id list, the second list len is 2, [videoid, imageid]
        class_id：int, the CLASSES index, 0 is background index, 1-19 is person part label index
        iou_threshold：list, for example IOU_THRE = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    Returns:
        AP: list, Average Precision of specific class

    """

    iou_thre_num = len(iou_threshold)
    ap = np.zeros((iou_thre_num,))

    gt_mask_num = 0  # record
    pre_mask_num = 0  #
    tp = []
    fp = []
    scores = []
    for i in range(iou_thre_num):
        tp.append([])
        fp.append([])

    print("process class", CLASSES[class_id], class_id)
    if os.path.exists(save_path):
        ap = np.loadtxt(save_path)
        print(CLASSES[class_id], "exists in", save_path)
        return ap

    for image_id in image_id_list:

        inst_part_gt = Image.open(os.path.join(INST_PART_GT_DIR, image_id[0], '%s.png' % image_id[1]))
        inst_part_gt = np.array(inst_part_gt)
        rfp = open(os.path.join(INST_PART_GT_DIR, image_id[0], '%s.txt' % image_id[1]), 'r')
        gt_part_id = []
        for line in rfp.readlines():
            line = line.strip().split(' ')
            gt_part_id.append([int(line[0]), int(line[1])])
        rfp.close()

        pre_img = Image.open(os.path.join(PREDICT_DIR, image_id[0], 'instance_parsing', '%s.png' % image_id[1]))
        pre_img = np.array(pre_img)
        rfp = open(os.path.join(PREDICT_DIR, image_id[0], 'instance_parsing', '%s.txt' % image_id[1]), 'r')
        items = [x.strip().split(' ') for x in rfp.readlines()]
        rfp.close()

        pre_id = []
        pre_scores = []
        for i in range(len(items)):
            if int(items[i][0]) == class_id:
                pre_id.append(i + 1)
                pre_scores.append(float(items[i][1]))

        gt_id = []
        for i in range(len(gt_part_id)):
            if gt_part_id[i][1] == class_id:
                gt_id.append(gt_part_id[i][0])
        # gt_mask: list, len n_gt_inst, the value is a ndarray,
        #   ndarray value is 0 and 1, 0 is bg, 1 represent that there are one person part from gt_id
        gt_mask, n_gt_inst = convert2evalformat(inst_part_gt, set(gt_id))
        pre_mask, n_pre_inst = convert2evalformat(pre_img, set(pre_id))

        gt_mask_num += n_gt_inst
        pre_mask_num += n_pre_inst

        if n_pre_inst == 0:
            continue

        scores += pre_scores

        if n_gt_inst == 0:
            for i in range(n_pre_inst):
                for k in range(iou_thre_num):
                    fp[k].append(1)
                    tp[k].append(0)
            continue

        gt_mask = np.stack(gt_mask)  # ndarray, shape [n_gt_inst, height, width]
        pre_mask = np.stack(pre_mask)  # ndarray, shape [n_pre_inst, height, width]
        # Compute IoU overlaps [pred_masks, gt_makss]
        overlaps = compute_mask_overlaps(pre_mask, gt_mask)  # shape [num_instances_pre, num_instances_gt]

        # print('overlaps.shape',overlaps.shape)

        max_overlap_ind = np.argmax(overlaps, axis=1)

        # l = len(overlaps[:,max_overlap_ind])
        for i in np.arange(len(max_overlap_ind)):  # pred inst number
            max_iou = overlaps[i][max_overlap_ind[i]]
            # print('max_iou :', max_iou)
            for k in range(iou_thre_num):
                if max_iou > iou_threshold[k]:
                    tp[k].append(1)
                    fp[k].append(0)
                else:
                    tp[k].append(0)
                    fp[k].append(1)

    ind = np.argsort(scores)[::-1]

    for k in range(iou_thre_num):
        m_tp = tp[k]
        m_fp = fp[k]
        m_tp = np.array(m_tp)
        m_fp = np.array(m_fp)

        m_tp = m_tp[ind]
        m_fp = m_fp[ind]

        m_tp = np.cumsum(m_tp)
        m_fp = np.cumsum(m_fp)
        # print('m_tp : ',m_tp)
        # print('m_fp : ', m_fp)
        recall = m_tp / (float(gt_mask_num) + 1e-7)
        precition = m_tp / np.maximum(m_fp + m_tp, np.finfo(np.float64).eps)

        # Compute mean AP over recall range
        ap[k] = voc_ap(recall, precition, False)
    np.savetxt(save_path, ap)
    return ap


if __name__ == '__main__':
    """
    # testing vip val dataset 2445 images spend almost 19 min 5 process
    command
    1: nohup python3 -u evaluate/evalute_inst_part_ap.py >> outs/eval_20190520a_epoch038.txt &
    2: tail -f outs/eval_20190520a_epoch038.txt
    Instance_ids:存储人和身体部位组合之后的结果，每个人的每个身体部位使用的不同的标签表示，该文件夹内主要用于评估evalute_inst_part_ap.py
    身体部位标签如下所示：
    (1, "hat")(2, "hair")(3, "gloves")(4, "sun-glasses")(5, "upper-clothes")(6, "dress")(7, "coat")(8, "socks")(9, "pants")(10, "torso-skin")
    (11, "scarf")(12, "skirt")(13, "face")(14, "left-arm")(15, "right-arm")(16, "left-leg")(17, "right-leg")(18, "left-shoe")(19, "right-shoe")
    
    图片和文件第一列中存储的是每个人每个身体部位的不同标签值
    
    对于txt文件中的三列：[注]预测是编号是连续的，和下面编号方式不一样
        第一列即为每个人每个身体部位的标签，由于身体部位有19个，0表示背景，从1开始编号，
            因此，第一个每个身体部位编号范围为1-19，第二个人编号范围为21-39，20是第二个人的背景，以此类推，第n个人编号范围为从20*n+1到20*n+19
        第二列为身体部位的标签，
        第三列为每个人的标签第一列即为每个人每个身体部位的标签，
    --videos45
        000000000001.png,000000000026.png,000000000051.png,......,000000000226.png  
        000000000001.txt,000000000026.txt,000000000051.txt,......,000000000226.txt  
            txt文件内容如下所示
            2 2 1
            5 5 1    #  5%20 == 5表示第一个人身体部位标签5
            14 14 1
            21 1 2   # 21%20 == 1，即表示第2个人的身体部位标签1
            22 2 2   # 22%20 == 2，即表示第2个人的身体部位标签2
    --videos86
        000000000001.png,000000000026.png,000000000051.png,......,000000000401.png
        000000000001.txt,000000000026.txt,000000000051.txt,......,000000000401.txt
    """
    print("result of", PREDICT_DIR)
    t0 = time.time()
    pool = multiprocessing.Pool(processes=NUM_PROCESS)
    image_list = []  # list, the value is also a list, len 2, [videoid, imageid]
    for vid in os.listdir(PREDICT_DIR):
        for img in os.listdir(os.path.join(PREDICT_DIR, vid, 'instance_parsing')):
            j = img.find('.')
            if img[j + 1:] == 'txt':
                image_list.append([vid, img[:j]])

    AP = np.zeros((len(CLASSES) - 1, len(IOU_THRE)))
    res = {}
    for ind in range(1, len(CLASSES)):
        t1 = time.time()
        print("start:", CLASSES[ind], ind)
        save_path = os.path.join(TMP_DIR, NAME, str(ind) + "_" + CLASSES[ind] + ".txt")
        a = pool.apply_async(compute_class_ap, args=(image_list, ind, IOU_THRE, save_path))
        res[ind - 1] = a
        print("eval", CLASSES[ind], "cost", time.time() - t1, "s")
    pool.close()
    pool.join()
    for r in sorted(res):
        # print("r", r)
        AP[r, :] = res[r].get()
    print("-----------------AP-----------------")
    np.savetxt(os.path.join(TMP_DIR, NAME, "AP.txt"), AP)
    print(AP)

    print("-----------------mAP-----------------")
    mAP = np.mean(AP, axis=0)
    mAP_path = os.path.join(TMP_DIR, NAME, "mAP.txt")
    np.savetxt(mAP_path, mAP)
    print(mAP)

    print("-----------------mAP_mean-----------------")
    mAP_mean = np.mean(mAP)
    with open(mAP_path, "a") as f:
        f.write("mAP:\n")
        f.write(str(mAP_mean))
    print(mAP_mean)
    print("total time:", time.time() - t0, "s")
