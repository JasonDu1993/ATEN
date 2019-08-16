# -*- coding: utf-8 -*-
# @Time    : 2019/4/19 10:13
# @Author  : Jason
# @Email   : 1358681631@qq.com
# @File    : triple_model.py.py
# @Software: PyCharm
import keras
import re
import datetime
import os
import numpy as np
import tensorflow as tf
from time import time

import keras.backend as K
from keras.layers import Input, Conv2D, AtrousConv2D, BatchNormalization, ReLU, Reshape, Activation
from keras.layers import Add, Deconv2D, Dense, Lambda, Concatenate, Multiply
from keras.layers import Conv3D, Deconv3D
from keras.layers import LSTM, ConvLSTM2D, ConvLSTM2DCell
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models.convolutional_recurrent import ConvGRU2D
from keras.models import Model
from keras.utils.vis_utils import plot_model

from configs.vip import VIPDataset
from configs.vip import VideoModelConfig
from utils import util

sess = tf.Session()


def Conv2D_BN_Relu(x, filters, kernel_size, name, padding, strides=(1, 1), use_bias=True):
    h_kernel, w_kernel = kernel_size
    x = Conv2D(filters, (h_kernel, w_kernel), strides=strides, padding=padding, name=name + "_conv2d",
               use_bias=use_bias)(x)
    x = BatchNormalization(name=name + "_bn")(x)
    x = ReLU(name=name + "_relu")(x)
    return x


def multi_input_Conv2D_BN_Relu(input_list, filters, kernel_size, name, padding, strides=(1, 1), use_bias=True):
    h_kernel, w_kernel = kernel_size
    conv = Conv2D(filters, (h_kernel, w_kernel), strides=strides, padding=padding, name=name + "_conv2d",
                  use_bias=use_bias)
    bn = BatchNormalization(name=name + "_bn")

    features = []
    for i, x in enumerate(input_list):
        x = conv(x)
        x = bn(x)
        x = ReLU(name=name + "_relu_" + str(i))(x)
        features.append(x)
    return features


def Atrous_Conv2D_BN_Relu(x, filters, kernel_size, name, padding, dilation_rate, strides=(1, 1), use_bias=True):
    k1, k2 = kernel_size
    x = Conv2D(filters, (k1, k2), strides, padding=padding, dilation_rate=dilation_rate, name=name + "_atrous_conv2d",
               use_bias=use_bias)(x)
    x = BatchNormalization(name=name + "_bn")(x)
    x = ReLU(name=name + "_relu")(x)
    return x


def multi_input_Atrous_Conv2D_BN_Relu(input_list, filters, kernel_size, name, padding, dilation_rate, strides=(1, 1),
                                      use_bias=True):
    h_kernel, w_kernel = kernel_size
    conv = Conv2D(filters, (h_kernel, w_kernel), strides=strides, padding=padding, dilation_rate=dilation_rate,
                  name=name + "_atrous_conv2d",
                  use_bias=use_bias)
    bn = BatchNormalization(name=name + "_bn")

    features = []
    for i, x in enumerate(input_list):
        x = conv(x)
        x = bn(x)
        x = ReLU(name=name + "_relu_" + str(i))(x)
        features.append(x)
    return features


def identity_block_share(input_list, kernel_size, filters, name):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    """
    filter1, filter2, filter3 = filters
    # input_filter = input_list[0].shape()
    # print(input_filter)
    # assert input_filter == filters

    conv1 = multi_input_Conv2D_BN_Relu(input_list, filter1, (1, 1), name + "_a", padding="same", strides=(1, 1))
    conv2 = multi_input_Conv2D_BN_Relu(conv1, filter2, (kernel_size, kernel_size), name + "_b", padding="same",
                                       strides=(1, 1))
    conv3 = multi_input_Conv2D_BN_Relu(conv2, filter3, (1, 1), name + "_c", padding="same", strides=(1, 1))

    features = []
    for i, (input_tensor, output_tensor) in enumerate(zip(input_list, conv3)):
        x = Add(name=name + "_add_" + str(i))([input_tensor, output_tensor])
        x = ReLU(name=name + "_relu_" + str(i))(x)
        features.append(x)
    return features


def identity_block_share(input_list, kernel_size, filters, name):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    """
    filter1, filter2, filter3 = filters
    # input_filter = input_list[0].shape()
    # print(input_filter)
    # assert input_filter == filters

    conv1 = multi_input_Conv2D_BN_Relu(input_list, filter1, (1, 1), name + "_a", padding="same", strides=(1, 1))
    conv2 = multi_input_Conv2D_BN_Relu(conv1, filter2, (kernel_size, kernel_size), name + "_b", padding="same",
                                       strides=(1, 1))
    conv3 = multi_input_Conv2D_BN_Relu(conv2, filter3, (1, 1), name + "_c", padding="same", strides=(1, 1))

    features = []
    for i, (input_tensor, output_tensor) in enumerate(zip(input_list, conv3)):
        x = Add(name=name + "_add_" + str(i))([input_tensor, output_tensor])
        x = ReLU(name=name + "_relu_" + str(i))(x)
        features.append(x)
    return features


def temporal_propagation(inputs_image_keys):
    x = multi_input_Conv2D_BN_Relu(inputs_image_keys, 256, kernel_size=(3, 3), name="tp_downsample", padding="valid",
                                   strides=(2, 2))
    x = identity_block_share(x, kernel_size=3, filters=[64, 64, 256], name="tp1")
    x = identity_block_share(x, kernel_size=3, filters=[64, 64, 256], name="tp2")
    x = identity_block_share(x, kernel_size=3, filters=[64, 64, 256], name="tp3")
    # x = K.stack(x, axis=1)  # 使用上面这个会报错AttributeError: 'NoneType' object has no attribute '_inbound_nodes'
    x = Lambda(lambda x: tf.stack(x, axis=1), name="tp_lambda_stack")(x)
    x = ConvGRU2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", dilation_rate=(1, 1),
                  name="tp_gru")(x)
    x = Conv2D_BN_Relu(x, 256, (3, 3), "tp_conv", padding="same")
    return x


def mask_propagation(inputs_mask_keys):
    x = multi_input_Conv2D_BN_Relu(inputs_mask_keys, 256, kernel_size=(3, 3), name="mp_downsample", padding="valid",
                                   strides=(2, 2))
    x = identity_block_share(x, kernel_size=3, filters=[64, 64, 256], name="mp1")
    x = identity_block_share(x, kernel_size=3, filters=[64, 64, 256], name="mp2")
    x = identity_block_share(x, kernel_size=3, filters=[64, 64, 256], name="mp3")
    # x = K.stack(x, axis=1)  # 使用上面这个会报错AttributeError: 'NoneType' object has no attribute '_inbound_nodes'
    x = Lambda(lambda x: tf.stack(x, axis=1), name="mp_lambda_stack")(x)
    x = ConvGRU2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", dilation_rate=(1, 1),
                  name="mp_gru")(x)
    x = Conv2D_BN_Relu(x, 256, (3, 3), "mp_conv", padding="same")
    return x


def triple_single_attention(inputs, inputs_image_keys, inputs_mask_keys, name):
    x = Conv2D_BN_Relu(inputs, 64, kernel_size=(3, 3), name=name + "_input_a", padding="same", strides=(1, 1))
    x1 = Conv2D_BN_Relu(inputs_image_keys, 64, kernel_size=(3, 3), name=name + "_images_a", padding="same",
                        strides=(1, 1))
    x2 = Conv2D_BN_Relu(inputs_mask_keys, 64, kernel_size=(3, 3), name=name + "_masks_a", padding="same",
                        strides=(1, 1))
    x1 = Add(name=name + "_image_add")([x, x1])
    x1 = ReLU(name=name + "_image_relu")(x1)

    x2 = Add(name=name + "_mask_add")([x, x2])
    x2 = ReLU(name=name + "_mask_relu")(x2)
    x1 = Conv2D_BN_Relu(x1, 64, kernel_size=(3, 3), name=name + "_image_b", padding="same", strides=(1, 1))
    x2 = Conv2D_BN_Relu(x2, 64, kernel_size=(3, 3), name=name + "_mask_b", padding="same", strides=(1, 1))
    f = Multiply(name=name + "_multiply")([x1, x2])
    x1 = Add(name=name + "_image_output_add1")([x1, f])
    x1 = ReLU(name=name + "_image_output_relu1")(x1)

    x2 = Add(name=name + "_image_output_add2")([x2, f])
    x2 = ReLU(name=name + "_image_output_relu2")(x2)

    x = Add(name=name + "_output_add")([x, f])
    x = ReLU(name=name + "_output_relu")(x)

    return x, x1, x2


def triple_multi_attention(inputs, inputs_image_keys, inputs_mask_keys, name):
    x = Conv2D_BN_Relu(inputs, 64, kernel_size=(3, 3), name=name + "_input_a", padding="same", strides=(1, 1))
    inputs_image_keys = multi_input_Conv2D_BN_Relu(inputs_image_keys, 64, kernel_size=(3, 3), name=name + "_images_a",
                                                   padding="same", strides=(1, 1))
    inputs_mask_keys = multi_input_Conv2D_BN_Relu(inputs_mask_keys, 64, kernel_size=(3, 3), name=name + "_masks_a",
                                                  padding="same", strides=(1, 1))

    f1 = []
    f2 = []
    for i, (x1, x2) in enumerate(zip(inputs_image_keys, inputs_mask_keys)):
        x1 = Add(name=name + "_image_add_" + str(i))([x, x1])
        x1 = ReLU(name=name + "_image_relu_" + str(i))(x1)
        f1.append(x1)
        x2 = Add(name=name + "_mask_add_" + str(i))([x, x2])
        x2 = ReLU(name=name + "_mask_relu_" + str(i))(x2)
        f2.append(x2)
    f1 = multi_input_Conv2D_BN_Relu(f1, 64, kernel_size=(3, 3), name=name + "_images_b", padding="same", strides=(1, 1))
    f2 = multi_input_Conv2D_BN_Relu(f2, 64, kernel_size=(3, 3), name=name + "_masks_b", padding="same", strides=(1, 1))
    res = []
    add = Add(name=name + "_output_add")
    relu = ReLU(name=name + "_output_relu")
    output_image_features = []
    output_mask_features = []
    for i, (f1_v, f2_v) in enumerate(zip(f1, f2)):
        f = Multiply(name=name + "_multiply_" + str(i))([f1_v, f2_v])
        f1_v = Add(name=name + "_image_output_add_" + str(i))([f1_v, f])
        f1_v = ReLU(name=name + "_image_output_relu_" + str(i))(f1_v)
        output_image_features.append(f1_v)
        f2_v = Add(name=name + "_mask_output_add_" + str(i))([f2_v, f])
        f2_v = ReLU(name=name + "_mask_output_relu_" + str(i))(f2_v)
        output_mask_features.append(f2_v)

        x = add([x, f])
        x = relu(x)
    outputs = x
    return outputs, output_image_features, output_mask_features


def temporal_and_mask_single_propagation(inputs, inputs_image_keys, inputs_mask_keys):
    x1 = temporal_propagation(inputs_image_keys)
    x2 = mask_propagation(inputs_mask_keys)
    x = Conv2D_BN_Relu(inputs, 256, kernel_size=(3, 3), name="input_downsample", padding="valid",
                       strides=(2, 2))
    x, x1, x2 = triple_single_attention(x, x1, x2, "tri_single_attention")
    x = Add(name="tmp_a")([x, x1, x2])
    x = Conv2D_BN_Relu(x, 256, (3, 3), "tmp_a", padding="same")
    return x


def temporal_and_mask_multi_propagation(inputs, inputs_image_keys, inputs_mask_keys):
    x1 = temporal_propagation(inputs_image_keys)
    x1_multi = []
    for i in range(len(inputs_image_keys)):
        x1_multi.append(ReLU(name="x1_" + str(i))(x1))

    x2 = mask_propagation(inputs_mask_keys)
    x2_multi = []
    for i in range(len(inputs_mask_keys)):
        x2_multi.append(ReLU(name="x2_" + str(i))(x2))
    x = Conv2D_BN_Relu(inputs, 256, kernel_size=(3, 3), name="input_downsample", padding="valid",
                       strides=(2, 2))
    x, x1, x2 = triple_multi_attention(x, x1_multi, x2_multi, name="tri_multi_attention")
    add = Add(name="TMPropp_add")
    relu = ReLU(name="TMProp_relu")
    res = []
    for f1, f2 in zip(x1, x2):
        x = add([f1, f2])
        x = relu(x)
        res.append(x)
    x = Add(name="TMProp_output_add")([x] + res)
    x = ReLU(name="TMProp_output_relu")(x)
    x = Conv2D_BN_Relu(x, 256, (3, 3), "output", padding="same")
    return x


def load_image_gt(dataset):
    image = dataset.load_image()
    dataset.


def data_generator(dataset, config, shuffle=True, augment=True, random_rois=0,
                   batch_size=1, detection_targets=False):
    """

    Args:

    Returns:

    """
    b = 0  # batch item index
    image_index = -1
    image_ids = np.copy(dataset.image_ids)
    error_count = 0
    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    anchors = util.generate_anchors(config.RPN_ANCHOR_SCALES,
                                    config.RPN_ANCHOR_RATIOS,
                                    config.BACKBONE_SHAPES[0],
                                    config.BACKBONE_STRIDES[0],
                                    config.RPN_ANCHOR_STRIDE)

    # Keras requires a generator to run indefinately.
    while True:
        try:
            # Increment index to pick next image. Shuffle if at the start of an epoch.
            image_index = (image_index + 1) % len(image_ids)
            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)

            # Get GT bounding boxes and masks for image.
            image_id = image_ids[image_index]
            image, key1, key2, key3, identity_ind, image_meta, gt_class_ids, gt_boxes, gt_masks, gt_parts = \
                load_image_gt(dataset, config, image_id, augment=augment,
                              use_mini_mask=config.USE_MINI_MASK)

            # Skip images that have no instances. This can happen in cases
            # where we train on a subset of classes and the image doesn't
            # have any of the classes we care about.
            if not np.any(gt_class_ids > 0):
                continue

            # RPN Targets
            rpn_match, rpn_bbox = build_rpn_targets(image.shape, anchors,
                                                    gt_class_ids, gt_boxes, config)

            # Mask R-CNN Targets
            if random_rois:
                rpn_rois = generate_random_rois(
                    image.shape, random_rois, gt_class_ids, gt_boxes)
                if detection_targets:
                    rois, mrcnn_class_ids, mrcnn_bbox, mrcnn_mask, mrcnn_part = \
                        build_detection_targets(
                            rpn_rois, gt_class_ids, gt_boxes, gt_masks, gt_parts, config)

            # Init batch arrays
            if b == 0:
                batch_image_meta = np.zeros(
                    (batch_size,) + image_meta.shape, dtype=image_meta.dtype)
                batch_rpn_match = np.zeros(
                    [batch_size, anchors.shape[0], 1], dtype=rpn_match.dtype)
                batch_rpn_bbox = np.zeros(
                    [batch_size, config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4], dtype=rpn_bbox.dtype)
                batch_images = np.zeros(
                    (batch_size,) + image.shape, dtype=np.float32)

                batch_key1s = np.zeros(
                    (batch_size,) + image.shape, dtype=np.float32)
                batch_key2s = np.zeros(
                    (batch_size,) + image.shape, dtype=np.float32)
                batch_key3s = np.zeros(
                    (batch_size,) + image.shape, dtype=np.float32)
                batch_identity_inds = np.zeros((batch_size, 1), dtype=np.int32)

                batch_gt_class_ids = np.zeros(
                    (batch_size, config.MAX_GT_INSTANCES), dtype=np.int32)
                batch_gt_boxes = np.zeros(
                    (batch_size, config.MAX_GT_INSTANCES, 4), dtype=np.int32)
                if config.USE_MINI_MASK:
                    batch_gt_masks = np.zeros((batch_size, config.MINI_MASK_SHAPE[0], config.MINI_MASK_SHAPE[1],
                                               config.MAX_GT_INSTANCES))
                else:
                    batch_gt_masks = np.zeros(
                        (batch_size, image.shape[0], image.shape[1], config.MAX_GT_INSTANCES))
                batch_gt_parts = np.zeros((batch_size, image.shape[0], image.shape[1]), dtype=np.uint8)
                if random_rois:
                    batch_rpn_rois = np.zeros(
                        (batch_size, rpn_rois.shape[0], 4), dtype=rpn_rois.dtype)
                    if detection_targets:
                        batch_rois = np.zeros(
                            (batch_size,) + rois.shape, dtype=rois.dtype)
                        batch_mrcnn_class_ids = np.zeros(
                            (batch_size,) + mrcnn_class_ids.shape, dtype=mrcnn_class_ids.dtype)
                        batch_mrcnn_bbox = np.zeros(
                            (batch_size,) + mrcnn_bbox.shape, dtype=mrcnn_bbox.dtype)
                        batch_mrcnn_mask = np.zeros(
                            (batch_size,) + mrcnn_mask.shape, dtype=mrcnn_mask.dtype)
                        batch_mrcnn_part = np.zeros(
                            (batch_size,) + mrcnn_part.shape, dtype=mrcnn_part.dtype)

            # If more instances than fits in the array, sub-sample from them.
            if gt_boxes.shape[0] > config.MAX_GT_INSTANCES:
                ids = np.random.choice(
                    np.arange(gt_boxes.shape[0]), config.MAX_GT_INSTANCES, replace=False)
                gt_class_ids = gt_class_ids[ids]
                gt_boxes = gt_boxes[ids]
                gt_masks = gt_masks[:, :, ids]

            # Add to batch
            batch_image_meta[b] = image_meta
            batch_rpn_match[b] = rpn_match[:, np.newaxis]
            batch_rpn_bbox[b] = rpn_bbox
            batch_images[b] = mold_image(image.astype(np.float32), config)
            batch_key1s[b] = mold_image(key1.astype(np.float32), config)
            batch_key2s[b] = mold_image(key2.astype(np.float32), config)
            batch_key3s[b] = mold_image(key3.astype(np.float32), config)
            batch_identity_inds[b, :] = identity_ind
            batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids
            batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
            batch_gt_masks[b, :, :, :gt_masks.shape[-1]] = gt_masks
            batch_gt_parts[b, :, :] = gt_parts
            if random_rois:
                batch_rpn_rois[b] = rpn_rois
                if detection_targets:
                    batch_rois[b] = rois
                    batch_mrcnn_class_ids[b] = mrcnn_class_ids
                    batch_mrcnn_bbox[b] = mrcnn_bbox
                    batch_mrcnn_mask[b] = mrcnn_mask
                    batch_mrcnn_part[b] = mrcnn_part
            b += 1

            # Batch full?
            if b >= batch_size:
                inputs = [batch_images, batch_key1s, batch_key2s, batch_key3s, batch_identity_inds,
                          batch_image_meta, batch_rpn_match, batch_rpn_bbox,
                          batch_gt_class_ids, batch_gt_boxes, batch_gt_masks, batch_gt_parts]
                outputs = []

                if random_rois:
                    inputs.extend([batch_rpn_rois])
                    if detection_targets:
                        inputs.extend([batch_rois])
                        # Keras requires that output and targets have the same number of dimensions
                        batch_mrcnn_class_ids = np.expand_dims(
                            batch_mrcnn_class_ids, -1)
                        outputs.extend(
                            [batch_mrcnn_class_ids, batch_mrcnn_bbox, batch_mrcnn_mask, batch_mrcnn_part])

                yield inputs, outputs

                # start a new batch
                b = 0
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # Log it and skip the image
            logging.exception("Error processing image {}".format(
                dataset.image_info[image_id]))
            error_count += 1
            if error_count > 5:
                raise


class TripleCNN(object):
    def __init__(self, mode, input_shape, config):
        assert mode in ['training', 'inference']
        self.mode = mode
        self.input_shape = input_shape
        self.config = config
        self.set_log_dir()
        self.keras_model = self.build_model(mode, input_shape, config)

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5
            regex = r".*/\w+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/aten\_\w+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                self.epoch = int(m.group(6)) + 1

        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "{}".format(self.config.NAME.lower()))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        # self.checkpoint_path = os.path.join(self.log_dir, "checkpoints",
        #                                     "parsing_rcnn_" + self.config.NAME.lower() +
        #                                     "_epoch{epoch:03d}_loss{loss:.3f}_valloss{val_loss:.3f}.h5")
        self.checkpoint_path = os.path.join(self.log_dir, "checkpoints",
                                            "aten_" + self.config.NAME.lower() +
                                            "_epoch{epoch:03d}_loss{loss:.3f}_valloss{val_loss:.3f}.h5")
        if not os.path.exists(os.path.dirname(self.checkpoint_path)):
            os.makedirs(os.path.dirname(self.checkpoint_path))

        self.log_path = os.path.join(self.log_dir, "logs", "aten_{}_{:%Y%m%dT%H%M}.csv".format(
            self.config.NAME.lower(), now))
        if not os.path.exists(os.path.dirname(self.log_path)):
            os.makedirs(os.path.dirname(self.log_path))

        self.tensorboard_dir = os.path.join(self.log_dir, "tensorboard")

    def build_model(self, input_shape, config=None, image_nums=5):
        if not config:
            config = VideoModelConfig()
        h, w, c = input_shape
        inputs = Input(shape=(h, w, c), name="input_image")
        inputs_image_keys = []
        inputs_mask_keys = []
        for i in range(image_nums):
            inputs_image_keys.append(Input(shape=(h, w, c), name="input_image_key" + str(i)))
            inputs_mask_keys.append(Input(shape=(h, w, c), name="input_mask_key" + str(i)))

        outputs = temporal_and_mask_single_propagation(inputs, inputs_image_keys, inputs_mask_keys)
        # outputs = temporal_and_mask_multi_propagation(inputs, inputs_image_keys, inputs_mask_keys)
        model = Model(inputs=[inputs] + inputs_image_keys + inputs_mask_keys, outputs=outputs, name="triple_model")
        # model = Model([inputs, inputs_image_keys, inputs_mask_keys], [outputs])
        print(model.summary())
        plot_model(model, "triple_single_model.png")
        # plot_model(model, "triple_multi_model.png")
        return model

    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers, period):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done alreay, so this actually determines
                the epochs to train in total rather than in this particaular
                call.
        layers: Allows selecting wich layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              heaads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
        """
        assert self.mode == "training", "Create model in training mode."
        # Pre-defined layer regular expressions
        layer_regex = {
            # all layers but the backbone
            "mask_heads": r"(mrcnn\_bbox\_.*)|(rpn\_.*)|(mrcnn\_class\_.*)|(mrcnn\_mask\_.*)|(mrcnn\_share\_.*)",
            "heads": r"(mrcnn\_.*)|(rpn\_.*)",
            # From a specific Resnet stage and up
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)",
            # All layers
            "all": ".*",
            "first_stage": r"(.*\_recurrent\_unit)|(mrcnn\_bbox\_.*)|(rpn\_.*)|(mrcnn\_class\_.*)|(mrcnn\_mask\_.*)|(mrcnn\_share\_.*)|(mrcnn\_global\_parsing\_c.*)",
            "second_stage": r"(.*\_recurrent\_unit)|(mrcnn\_.*)|(rpn\_.*)|(flownet\_.*)",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Data generators
        train_generator = data_generator(train_dataset, self.config, shuffle=True,
                                         batch_size=self.config.BATCH_SIZE)
        val_generator = data_generator(val_dataset, self.config, shuffle=True,
                                       batch_size=self.config.BATCH_SIZE)

        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.tensorboard_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            # keras.callbacks.ModelCheckpoint(self.checkpoint_path, period=period,
            #                                 verbose=0, save_weights_only=True),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path, verbose=0, save_best_only=True,
                                            save_weights_only=True),
            keras.callbacks.CSVLogger(self.log_path),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1),
            keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=12, verbose=1),
        ]

        # Train
        log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=next(val_generator),
            validation_steps=self.config.VALIDATION_STEPS,
            # max_queue_size=100,
            # workers=max(self.config.BATCH_SIZE // 2, 2),
            # use_multiprocessing=True,
            verbose=1,
        )
        self.epoch = max(self.epoch, epochs)


if __name__ == '__main__':
    TripleCNN(mode="training", input_shape=(512, 512, 256))
