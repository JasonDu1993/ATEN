import datetime
import logging
import os
import random
import re
import math
# Requires TensorFlow 1.3+ and Keras 2.0.8+.
from distutils.version import LooseVersion

import keras
import keras.backend as K
import keras.engine as KE
import keras.layers as KL
import keras.models as KM
import numpy as np
import skimage.transform
import tensorflow as tf

from keras.utils.vis_utils import plot_model
from utils import util

assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')
from configs.vipdataset_for_mfp import ParsingRCNNModelConfig


############################################################
#  config
############################################################
class MFPConfig(ParsingRCNNModelConfig):
    IS_PRE_IMAGE = False
    IS_PRE_MASK = True
    IS_PRE_PART = True

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 450  # 450, 256
    IMAGE_MAX_DIM = 512  # 512, 416， 384（16*24）
    # use small pre image for training
    PRE_IMAGE_SHAPE = [128, 128, 3]  # needed 128(PRE_IMAGE_SHAPE[0]) * 4 = 512(IMAGE_MAX_DIM)

    PRE_MULTI_FRAMES = 3
    RECURRENT_UNIT = "gru"
    assert RECURRENT_UNIT in ["gru", "lstm"]
    RECURRENT_FILTER = 64
    USE_RPN_ROIS = True  # for rpn


############################################################
#  Utility Functions
############################################################

def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  min: {:10.5f}  max: {:10.5f}".format(
            str(array.shape),
            array.min() if array.size else "",
            array.max() if array.size else ""))
    print(text)


class BatchNorm(KL.BatchNormalization):
    """Batch Normalization class. Subclasses the Keras BN class and
    hardcodes training=False so the BN layer doesn't update
    during training.

    Batch normalization has a negative effect on training if batches are small
    so we disable it here.
    """

    def __init__(self, training=False, **kwargs):
        super(BatchNorm, self).__init__(**kwargs)
        self.training = training

    def call(self, inputs, training=None):
        return super(self.__class__, self).call(inputs, training=self.training)


def identity_block(input_tensor, kernel_size, filters, stage, block,
                   use_bias=True, roi_res=False):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    if roi_res:
        conv_name_base = 'mrcnn_class_' + conv_name_base
        bn_name_base = 'mrcnn_class_' + bn_name_base

    x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                  use_bias=use_bias)(input_tensor)
    x = BatchNorm(axis=-1, name=bn_name_base + '2a')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(axis=-1, name=bn_name_base + '2b')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                  use_bias=use_bias)(x)
    x = BatchNorm(axis=-1, name=bn_name_base + '2c')(x)

    x = KL.Add()([x, input_tensor])
    x = KL.Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2), use_bias=True, roi_res=False):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    if roi_res:
        conv_name_base = 'mrcnn_class_' + conv_name_base
        bn_name_base = 'mrcnn_class_' + bn_name_base

    x = KL.Conv2D(nb_filter1, (1, 1), strides=strides,
                  name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = BatchNorm(axis=-1, name=bn_name_base + '2a')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(axis=-1, name=bn_name_base + '2b')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                  use_bias=use_bias)(x)
    x = BatchNorm(axis=-1, name=bn_name_base + '2c')(x)

    shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides,
                         name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    shortcut = BatchNorm(axis=-1, name=bn_name_base + '1')(shortcut)

    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu')(x)
    return x


# Atrous-Convolution version of residual blocks
def atrous_identity_block(input_tensor, kernel_size, filters, stage,
                          block, atrous_rate=(2, 2), use_bias=True, roi_res=False):
    '''The identity_block is the block that has no conv layer at shortcut
    # Arguments
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    if roi_res:
        conv_name_base = 'mrcnn_mask_' + conv_name_base
        bn_name_base = 'mrcnn_mask_' + bn_name_base

    x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                  use_bias=use_bias)(input_tensor)
    x = BatchNorm(axis=-1, name=bn_name_base + '2a')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), dilation_rate=atrous_rate,
                  padding='same', name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(axis=-1, name=bn_name_base + '2b')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                  use_bias=use_bias)(x)
    x = BatchNorm(axis=-1, name=bn_name_base + '2c')(x)

    x = KL.Add()([x, input_tensor])
    x = KL.Activation('relu')(x)
    return x


def atrous_conv_block(input_tensor, kernel_size, filters, stage,
                      block, strides=(1, 1), atrous_rate=(2, 2), use_bias=True, roi_res=False):
    '''conv_block is the block that has a conv layer at shortcut
    # Arguments
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    if roi_res:
        conv_name_base = 'mrcnn_mask_' + conv_name_base
        bn_name_base = 'mrcnn_mask_' + bn_name_base

    x = KL.Conv2D(nb_filter1, (1, 1), strides=strides,
                  name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = BatchNorm(axis=-1, name=bn_name_base + '2a')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', dilation_rate=atrous_rate,
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(axis=-1, name=bn_name_base + '2b')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                  use_bias=use_bias)(x)
    x = BatchNorm(axis=-1, name=bn_name_base + '2c')(x)

    shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides,
                         name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    shortcut = BatchNorm(axis=-1, name=bn_name_base + '1')(shortcut)

    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu')(x)
    return x


def conv3d_gru2d_unit(temporal_features, filter, name, kernel_size=(3, 3), padding="same"):
    input_tensor = KL.Lambda(lambda x: tf.stack(x, axis=1, name=name + "_stack"))(temporal_features)
    from models.convolutional_recurrent import ConvGRU2D
    feature_conv3d = KL.Conv3D(filter, (3, 3, 3), padding="same", activation="relu", name=name + "_conv3d")(
        input_tensor)
    initial_state = K.mean(feature_conv3d, axis=1)
    x = ConvGRU2D(filters=filter, kernel_size=kernel_size, name=name,
                  padding=padding, return_sequences=False)(input_tensor, initial_state=initial_state)
    return x


def conv_lstm_unit(temporal_features, filter, name, kernel_size=(3, 3), padding="same", initial_state=None):
    input_tensor = KL.Lambda(lambda x: tf.stack(x, axis=1))(temporal_features)
    x = KL.ConvLSTM2D(filters=filter, kernel_size=kernel_size, name=name,
                      padding=padding, return_sequences=False)(input_tensor, initial_state=initial_state)
    return x


def mean_tensor(x):
    assert isinstance(x, list), "need a list for input"
    l = len(x)
    x = KL.Add()(x)
    x = x / l
    return x


def deeplab_resnet(img_input, architecture):
    """
    Build the architecture of resnet
    img_input:Tensor("input_image:0", shape=(?, 512, 512, 3), dtype=float32)
    architecture:Str, "resnet50" or "resnet101"
    Rreturn:
        C1:Tensor("max_pooling2d_1/MaxPool:0", shape=(?, 128, 128, 64), dtype=float32)
        C2:Tensor("activation_10/Relu:0", shape=(?, 128, 128, 256), dtype=float32)
        C3:Tensor("activation_22/Relu:0", shape=(?, 64, 64, 512), dtype=float32)
        C4:Tensor("activation_40/Relu:0", shape=(?, 32, 32, 1024), dtype=float32)
        C5:Tensor("activation_49/Relu:0", shape=(?, 32, 32, 2048), dtype=float32)
    """

    # Stage 1
    x = KL.ZeroPadding2D((3, 3))(img_input)
    x = KL.Conv2D(64, (7, 7), strides=(2, 2),
                  name='conv1', use_bias=False)(x)
    x = BatchNorm(axis=-1, name='bn_conv1')(x)
    x = KL.Activation('relu')(x)
    C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    # Stage 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), use_bias=False)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', use_bias=False)
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', use_bias=False)
    # Stage 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', use_bias=False)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b1', use_bias=False)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b2', use_bias=False)
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='b3', use_bias=False)
    # Stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', use_bias=False)
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(1, block_count + 1):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b%d' % i, use_bias=False)
    C4 = x
    # Stage 5
    x = atrous_conv_block(x, 3, [512, 512, 2048], stage=5, block='a', atrous_rate=(2, 2), use_bias=False)
    x = atrous_identity_block(x, 3, [512, 512, 2048], stage=5, block='b', atrous_rate=(2, 2), use_bias=False)
    C5 = x = atrous_identity_block(x, 3, [512, 512, 2048], stage=5, block='c', atrous_rate=(2, 2), use_bias=False)

    return [C1, C2, C3, C4, C5]


############################################################
#  Proposal Layer
############################################################

def apply_box_deltas_graph(anchors, deltas):
    """Applies the given deltas to the given anchors.
    Args:
        anchors: Tensor, shape=(N, 4) where each row is y1, x1, y2, x2
        deltas: Tensor, shape=(N, 4) where each row is [dy, dx, log(dh), log(dw)]
    Returns:
        result: Tensor, shape=(N, 4) where each row is y1, x1, y2, x2
    """
    # Convert to y, x, h, w
    height = anchors[:, 2] - anchors[:, 0]
    width = anchors[:, 3] - anchors[:, 1]
    center_y = anchors[:, 0] + 0.5 * height
    center_x = anchors[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= tf.exp(deltas[:, 2])
    width *= tf.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")
    return result


def clip_boxes_graph(boxes, window):
    """去除预测出来的boxes坐标限制在window所在范围内
    boxes: numpy.ndarray, [N=pre_nms_limit=12000, 4] each row is y1, x1, y2, x2
    window: Tensor, [4] in the form y1, x1, y2, x2
    """
    # Split corners
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
    # Clip
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)  # shape (N=12000, 1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")  # shape (N=12000, 4)
    return clipped


class ProposalLayer(KE.Layer):
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinement details to anchors.

    Inputs:
        rpn_class: [batch, anchors_num=128*128*5*3, (bg prob, fg prob)]
        rpn_bbox: [batch, anchors_num=128*128*5*3, (dy, dx, log(dh), log(dw))]

    Returns:
        Proposals in normalized coordinates [batch, proposal_count, (y1, x1, y2, x2)]
    """

    def __init__(self, proposal_count, pre_proposal_count, nms_threshold, anchors,
                 config=None, **kwargs):
        """
        Args:
            proposal_count:training default 2000, inference default 1000
            pre_proposal_count:training default 12000, inference default 6000
            nms_threshold: default 0.7
            anchors: [anchors_num=128*128*5*3, (y1, x1, y2, x2)] anchors defined in image coordinates
        """
        super(ProposalLayer, self).__init__(**kwargs)
        self.config = config
        self.proposal_count = proposal_count
        self.pre_proposal_count = pre_proposal_count
        self.nms_threshold = nms_threshold
        self.anchors = anchors.astype(np.float32)

    def call(self, inputs):  #
        """ Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]

        Args:
            inputs: list: [<tf.Tensor 'rpn_class_xxx/truediv:0' shape=(batch, 245760=128*128*3*5, 2) dtype=float32>,
                            <tf.Tensor 'lambda_63/Reshape:0' shape=(batch, 245760=128*128*3*5, 4) dtype=float32>]
        Returns:
            proposals: [batch, proposal_count, 4], the coordinate is normalized to 1

        """
        scores = inputs[0][:, :, 1]  # shape (batch, 245760=128*128*5*3)
        # Box deltas [batch, num_rois, 4]
        deltas = inputs[1]  # shape (batch, 245760, 4)
        deltas = deltas * np.reshape(self.config.RPN_BBOX_STD_DEV, [1, 1, 4])  # shape (batch, 245760, 4)
        # Base anchors
        anchors = self.anchors  # shape (245760, 4)

        # Improve performance by trimming to top anchors by score
        # and doing the rest on the smaller subset.
        pre_nms_limit = min(self.pre_proposal_count, self.anchors.shape[0])
        ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True,
                         name="top_anchors").indices  # (batch, pre_nms_limit)
        scores = util.batch_slice([scores, ix], lambda x, y: tf.gather(x, y),
                                  self.config.IMAGES_PER_GPU)  # shape(batch, pre_nms_limit)
        deltas = util.batch_slice([deltas, ix], lambda x, y: tf.gather(x, y),
                                  self.config.IMAGES_PER_GPU)  # shape(batch, pre_nms_limit, 4)
        anchors = util.batch_slice(ix, lambda x: tf.gather(anchors, x),
                                   self.config.IMAGES_PER_GPU,
                                   names=["pre_nms_anchors"])  # shape(batch, pre_nms_limit, 4)

        # Apply deltas to anchors to get refined anchors.
        # [batch, pre_nms_limit, (y1, x1, y2, x2)]
        boxes = util.batch_slice([anchors, deltas],
                                 lambda x, y: apply_box_deltas_graph(x, y),
                                 self.config.IMAGES_PER_GPU,
                                 names=["refined_anchors"])  # shape(batch, pre_nms_limit, 4)

        # Clip to image boundaries. [batch, pre_nms_limit, (y1, x1, y2, x2)]
        height, width = self.config.IMAGE_SHAPE[:2]  # [512, 512]
        window = np.array([0, 0, height, width]).astype(np.float32)
        boxes = util.batch_slice(boxes,
                                 lambda x: clip_boxes_graph(x, window),
                                 self.config.IMAGES_PER_GPU,
                                 names=["refined_anchors_clipped"])  # shape (batch, 12000, 4)

        # Filter out small boxes
        # According to Xinlei Chen's paper, this reduces detection accuracy
        # for small objects, so we're skipping it.

        # Normalize dimensions to range of 0 to 1.
        normalized_boxes = boxes / np.array([[height, width, height, width]])

        # Non-max suppression
        def nms(normalized_boxes, scores):
            indices = tf.image.non_max_suppression(
                normalized_boxes, scores, self.proposal_count,
                self.nms_threshold, name="rpn_non_max_suppression")
            proposals = tf.gather(normalized_boxes, indices)
            # Pad if needed
            padding = tf.maximum(self.proposal_count - tf.shape(proposals)[0], 0)
            proposals = tf.pad(proposals, [(0, padding), (0, 0)])
            return proposals

        proposals = util.batch_slice([normalized_boxes, scores], nms,
                                     self.config.IMAGES_PER_GPU)
        return proposals

    def compute_output_shape(self, input_shape):
        return (None, self.proposal_count, 4)


############################################################
#  ROIAlign Layer
############################################################

def log2_graph(x):
    """Implementatin of Log2. TF doesn't have a native implemenation."""
    return tf.log(x) / tf.log(2.0)


def roi_crop_and_resize(image, boxes, box_ind, crop_shape):
    """
    Better-aligned version of tf.image.crop_and_resize, following our definition of floating point boxes.

    Args:
        image: BHWC
        boxes: nx4, y1x1y2x2 normalized to 1
        box_ind: (n,)
        crop_shape (int,int), for example (14, 14)
    Returns:
        n,C,size,size
    """

    def transform_fpcoor_for_tf(boxes, image_shape, crop_shape):
        """
        The way tf.image.crop_and_resize works (with normalized box):
        Initial point (the value of output[0]): x0_box * (W_img - 1)
        Spacing: w_box * (W_img - 1) / (W_crop - 1)
        Use the above grid to bilinear sample.

        However, what we want is (with fpcoor box):
        Spacing: w_box / W_crop
        Initial point: x0_box + spacing/2 - 0.5
        (-0.5 because bilinear sample assumes floating point coordinate (0.0, 0.0) is the same as pixel value (0, 0))

        This function transform fpcoor boxes to a format to be used by tf.image.crop_and_resize

        Returns:
            [n, (y1,x1,y2,x2)], normalize to 1
        """
        y0, x0, y1, x1 = tf.split(boxes, 4, axis=1)

        # each bin shape
        spacing_w = (x1 - x0) / tf.to_float(crop_shape[1])
        spacing_h = (y1 - y0) / tf.to_float(crop_shape[0])

        nx0 = (x0 + spacing_w / 2 - 0.5) / tf.to_float(image_shape[1] - 1)
        ny0 = (y0 + spacing_h / 2 - 0.5) / tf.to_float(image_shape[0] - 1)

        nw = spacing_w * tf.to_float(crop_shape[1] - 1) / tf.to_float(image_shape[1] - 1)
        nh = spacing_h * tf.to_float(crop_shape[0] - 1) / tf.to_float(image_shape[0] - 1)
        ny1 = ny0 + nh
        nx1 = nx0 + nw
        return tf.concat([ny0, nx0, ny1, nx1], axis=1)

    image_shape = tf.shape(image)[1:3]  # [128, 128]
    boxes = boxes * tf.cast(tf.stack([image_shape[0], image_shape[1], image_shape[0], image_shape[1]]), tf.float32)
    boxes = transform_fpcoor_for_tf(boxes, image_shape, crop_shape)
    ret = tf.image.crop_and_resize(image, boxes, box_ind, crop_shape)
    return ret


def roi_align_graph(featuremap, boxes, box_inds, output_shape):
    """
    Args:
        featuremap: [batch, H=128, W=128, C=256]
        boxes: [N=batch * num_boxes, (y1, x1, y2, x2)] normalized coordinates
        box_inds: A 1-D int32 tensor of shape [N] with value in [0, batch-1]
        output_shape: A 1-D tensor of 2 elements, size = [crop_height=7, crop_width=7]

    Returns:
        [N=batch * num_boxes, pool_height=7, pool_width=7, C=256]
    """
    # sample 4 locations per roi bin
    # output [num_boxes, crop_height, crop_width, c]
    ret = roi_crop_and_resize(
        featuremap, boxes,
        box_inds,
        [output_shape[0] * 2, output_shape[1] * 2])
    ret = tf.nn.avg_pool(ret, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    return ret


class PyramidROIAlign(KE.Layer):
    """Implements ROI Pooling on multiple levels of the feature pyramid.

    Params:
    - pool_shape: [height, width] of the output pooled regions. Usually [7, 7]
    - image_shape: [height, width, chanells]. Shape of input image in pixels

    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates. Possibly padded with zeros if not enough
             boxes to fill the array.
    - Feature maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, height, width, channels]

    Output:
    Pooled regions in the shape: [batch, num_boxes, height, width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """

    def __init__(self, pool_shape, image_shape, **kwargs):
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)
        self.image_shape = tuple(image_shape)

    def call(self, inputs):
        # Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
        boxes = inputs[0]

        # Feature Maps. List of feature maps from different level of the
        # feature pyramid. Each is [batch, height, width, channels]
        feature_maps = inputs[1:]

        # Assign each ROI to a level in the pyramid based on the ROI area.
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        h = y2 - y1
        w = x2 - x1
        # Equation 1 in the Feature Pyramid Networks paper. Account for
        # the fact that our coordinates are normalized here.
        # e.g. a 224x224 ROI (in pixels) maps to P4
        image_area = tf.cast(
            self.image_shape[0] * self.image_shape[1], tf.float32)
        roi_level = log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
        roi_level = tf.minimum(5, tf.maximum(
            2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
        roi_level = tf.squeeze(roi_level, 2)

        # Loop through levels and apply ROI pooling to each. P2 to P5.
        pooled = []
        box_to_level = []
        for i, level in enumerate(range(2, 6)):
            ix = tf.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(boxes, ix)

            # Box indicies for crop_and_resize.
            box_indices = tf.cast(ix[:, 0], tf.int32)

            # Keep track of which box is mapped to which level
            box_to_level.append(ix)

            # Stop gradient propogation to ROI proposals
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

            # Crop and Resize
            pooled = roi_align_graph(
                feature_maps[i], level_boxes, box_indices, self.pool_shape)

        # Pack pooled features into one tensor
        pooled = tf.concat(pooled, axis=0)

        # Pack box_to_level mapping into one array and add another
        # column representing the order of pooled boxes
        box_to_level = tf.concat(box_to_level, axis=0)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
                                 axis=1)

        # Rearrange pooled features to match the order of the original boxes
        # Sort box_to_level by batch then box index
        # TF doesn't have a way to sort by two columns, so merge them and sort.
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
            box_to_level)[0]).indices[::-1]
        ix = tf.gather(box_to_level[:, 2], ix)
        pooled = tf.gather(pooled, ix)

        # Re-add the batch dimension
        pooled = tf.expand_dims(pooled, 0)
        return pooled

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + self.pool_shape + (input_shape[1][-1],)


class ROIAlign(KE.Layer):
    """Implements ROI Pooling on multiple levels of the feature pyramid.

    Params:
    - pool_shape: [pool_height, pool_width] of the output pooled regions. Usually [7, 7]
    - image_shape: [f_height=128, f_width=128, f_chanells=256]. Shape of input image in pixels

    Inputs:
    - boxes: [batch, num_boxes=128, (y1, x1, y2, x2)] in normalized
             coordinates. Possibly padded with zeros if not enough
             boxes to fill the array.
    - Feature maps: [batch, height, width, channels]

    Output:
    Pooled regions in the shape: [batch, num_boxes, pool_height=7, pool_width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """

    def __init__(self, pool_shape, image_shape, **kwargs):
        super(ROIAlign, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)
        self.image_shape = tuple(image_shape)

    def call(self, inputs):
        # Crop boxes [batch, num_boxes=128, (y1, x1, y2, x2)] in normalized coords
        boxes = inputs[0]

        # Feature Maps. List of feature maps from different level of the
        # feature pyramid. Each is [batch, f_height=128, f_width=128, f_channels=256]
        feature_map = inputs[1]

        assign = tf.cast(tf.ones(tf.shape(boxes)[0:2]), tf.int32)  # shape [batch, num_boxes]
        assign_ids = tf.where(tf.equal(assign, 1))  # shape [batch * num_boxes, 2], because assign all elements are 1

        new_boxes = tf.gather_nd(boxes, assign_ids)  # [batch * num_boxes, (y1, x1, y2, x2)] in normalized coords
        box_indices = tf.cast(assign_ids[:, 0], tf.int32)  # shape [batch * num_boxes, ]

        new_boxes = tf.stop_gradient(new_boxes)
        box_indices = tf.stop_gradient(box_indices)

        # Crop and Resize
        pooled = roi_align_graph(
            feature_map, new_boxes, box_indices, self.pool_shape)  # shape [batch * num_boxes, 7, 7, 256]
        # Re-add the batch dimension
        pooled = tf.expand_dims(pooled, 0)  # shape [1, batch * num_boxes, 7, 7, 256]
        return pooled

    def compute_output_shape(self, input_shape):
        return input_shape[0][0:2] + self.pool_shape + (input_shape[1][-1],)  # [batch, num_boxes=128, 7, 7, 256]


############################################################
#  Detection Target Layer
############################################################

def overlaps_graph(proposals, gt_boxes):
    """Computes IoU overlaps between two sets of boxes.
    proposals, gt_boxes: [N1, (y1, x1, y2, x2)], [N2, (y1,x1,y2,x2)].overlaps shape [N1, N2]
    """
    # 1. Tile gt_boxes and repeate proposals. This allows us to compare
    # every proposals against every gt_boxes without loops.
    # TF doesn't have an equivalent to np.repeate() so simulate it
    # using tf.tile() and tf.reshape.
    b1 = tf.reshape(tf.tile(tf.expand_dims(proposals, 1),
                            [1, 1, tf.shape(gt_boxes)[0]]), [-1, 4])
    b2 = tf.tile(gt_boxes, [tf.shape(proposals)[0], 1])
    # 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    # 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection
    # 4. Compute IoU and reshape to [proposals, gt_boxes]
    iou = intersection / union
    overlaps = tf.reshape(iou, [tf.shape(proposals)[0], tf.shape(gt_boxes)[0]])
    return overlaps


def detection_targets_graph(proposals, gt_class_ids, gt_boxes, gt_masks, config):
    """Generates detection targets for one image. Subsamples proposals and
    generates target class IDs, bounding box deltas, and masks for each.

    Args:
        proposals: [N=proposal_count=2000, (y1, x1, y2, x2)] in normalized coordinates. Might
                   be zero padded if there are not enough proposals.
        gt_class_ids: [MAX_GT_INSTANCES] int class IDs
        gt_boxes: [MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized coordinates.
        gt_masks: [height=56, width=56, MAX_GT_INSTANCES] of boolean type.

    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks. TRAIN_ROIS_PER_IMAGE default 200, in vip is 128
        rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized coordinates
        class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs. Zero padded.
        deltas: [TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw))]
                Class-specific bbox refinments.
        masks: [TRAIN_ROIS_PER_IMAGE, m_height=28, m_width=28). Masks cropped to bbox
               boundaries and resized to neural network output size.

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """
    # Assertions
    asserts = [
        tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals],
                  name="roi_assertion"),
    ]
    with tf.control_dependencies(asserts):
        proposals = tf.identity(proposals)

    # Remove zero padding
    proposals, _ = trim_zeros_graph(proposals, name="trim_proposals")
    gt_boxes, non_zeros = trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
    gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros,
                                   name="trim_gt_class_ids")
    gt_masks = tf.gather(gt_masks, tf.where(non_zeros)[:, 0], axis=2,
                         name="trim_gt_masks")

    # Compute overlaps matrix [proposals, gt_boxes]
    overlaps = overlaps_graph(proposals, gt_boxes)  # overlaps shape [tf.shape(proposals)[0], tf.shape(gt_boxes)[0]]

    # Determine postive and negative ROIs
    roi_iou_max = tf.reduce_max(overlaps, axis=1)
    # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
    positive_roi_bool = (roi_iou_max >= 0.5)
    positive_indices = tf.where(positive_roi_bool)[:, 0]

    # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
    negative_indices = tf.where(roi_iou_max < 0.5)[:, 0]

    # Subsample ROIs. Aim for 33% positive
    # Positive ROIs
    positive_count = int(config.TRAIN_ROIS_PER_IMAGE *
                         config.ROI_POSITIVE_RATIO)  # int(128*0.33)等于42
    positive_indices = tf.random_shuffle(positive_indices)[:positive_count]
    positive_count = tf.shape(positive_indices)[0]
    # Negative ROIs. Add enough to maintain positive:negative ratio.
    r = 1.0 / config.ROI_POSITIVE_RATIO  # r=1/0.33等于3.0303...
    negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count  # 3.0303 * 42 - 42 =85
    negative_indices = tf.random_shuffle(negative_indices)[:negative_count]  # int(1/0.33 * 42) - 42 等于127 - 42 = 85
    # Gather selected ROIs
    positive_rois = tf.gather(proposals, positive_indices)  # shape (positive_count, 4)
    negative_rois = tf.gather(proposals, negative_indices)  # shape (negative_count, 4)

    # Assign positive ROIs to GT boxes.
    positive_overlaps = tf.gather(overlaps, positive_indices)  # shape (positive_count, tf.shape(gt_boxes)[0])
    roi_gt_box_assignment = tf.argmax(positive_overlaps, axis=1)  # shape (positive_count,)
    roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)  # shape (positive_count, 4)
    roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)  # shape (positive_count,)

    # Compute bbox refinement for positive ROIs
    deltas = util.box_refinement_graph(positive_rois, roi_gt_boxes)
    deltas /= config.BBOX_STD_DEV

    # Assign positive ROIs to GT masks
    # Permute masks to [N, height, width, 1]
    transposed_masks = tf.expand_dims(tf.transpose(gt_masks, [2, 0, 1]), -1)  # shape (MAX_GT_INSTANCES, 56, 56,depth=1)
    # Pick the right mask for each ROI
    roi_masks = tf.gather(transposed_masks, roi_gt_box_assignment)  # shape (positive_count, 56, 56, 1)

    # Compute mask targets
    boxes = positive_rois
    if config.USE_MINI_MASK:
        # Transform ROI corrdinates from normalized image space
        # to normalized mini-mask space.
        y1, x1, y2, x2 = tf.split(positive_rois, 4, axis=1)
        gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(roi_gt_boxes, 4, axis=1)
        gt_h = gt_y2 - gt_y1
        gt_w = gt_x2 - gt_x1
        y1 = (y1 - gt_y1) / gt_h
        x1 = (x1 - gt_x1) / gt_w
        y2 = (y2 - gt_y1) / gt_h
        x2 = (x2 - gt_x1) / gt_w
        boxes = tf.concat([y1, x1, y2, x2], 1)
    box_ids = tf.range(0, tf.shape(roi_masks)[0])
    masks = tf.image.crop_and_resize(tf.cast(roi_masks, tf.float32), boxes,
                                     box_ids,
                                     config.MASK_SHAPE)  # shape (positive_count, 28, 28, 1)
    # Remove the extra dimension from masks.
    masks = tf.squeeze(masks, axis=3)  # shape (positive_count, 28, 28)

    # Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
    # binary cross entropy loss.
    masks = tf.round(masks)

    # Append negative ROIs and pad bbox deltas and masks that
    # are not used for negative ROIs with zeros.
    rois = tf.concat([positive_rois, negative_rois], axis=0)  # shape [127=(1/0.33* 42), 4]
    N = tf.shape(negative_rois)[0]
    P = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0)
    rois = tf.pad(rois, [(0, P), (0, 0)])
    roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, N + P), (0, 0)])  # shape [config.TRAIN_ROIS_PER_IMAGE, 4]
    roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])  # shape [config.TRAIN_ROIS_PER_IMAGE, ]
    deltas = tf.pad(deltas, [(0, N + P), (0, 0)])  # shape [config.TRAIN_ROIS_PER_IMAGE, 4]
    masks = tf.pad(masks, [[0, N + P], (0, 0), (0, 0)])  # shape [config.TRAIN_ROIS_PER_IMAGE, 28, 28]

    return rois, roi_gt_class_ids, deltas, masks


class DetectionTargetLayer(KE.Layer):
    """Subsamples proposals and generates target box refinment, class_ids,
    and masks for each.

    Args:
        proposals: [batch, N=proposal_count=2000, (y1, x1, y2, x2)] in normalized coordinates to 1. Might
                   be zero padded if there are not enough proposals.
        gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
        gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized
                  coordinates.
        gt_masks: [batch, height=56, width=56, MAX_GT_INSTANCES] of boolean type

    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
        rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized
              coordinates, TRAIN_ROIS_PER_IMAGE default 200, in vip is 128
        target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
        target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw))]
                       Class-specific bbox refinements.
        target_mask: [batch, TRAIN_ROIS_PER_IMAGE, m_height, m_width)
                     Masks cropped to bbox boundaries and resized to neural
                     network output size.

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """

    def __init__(self, config, **kwargs):
        super(DetectionTargetLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        proposals = inputs[0]  # [batch, N=proposal_count=2000, (y1, x1, y2, x2)] in normalized coordinates to 1
        gt_class_ids = inputs[1]  # [batch, MAX_GT_INSTANCES] Integer class IDs.
        gt_boxes = inputs[2]  # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized to 1
        gt_masks = inputs[3]  # [batch, height=56, width=56, MAX_GT_INSTANCES]

        # Slice the batch and run a graph for each slice
        # TODO: Rename target_bbox to target_deltas for clarity
        names = ["rois", "target_class_ids", "target_bbox", "target_mask"]
        outputs = util.batch_slice(
            [proposals, gt_class_ids, gt_boxes, gt_masks],
            lambda w, x, y, z: detection_targets_graph(
                w, x, y, z, self.config),
            self.config.IMAGES_PER_GPU, names=names)
        return outputs

    def compute_output_shape(self, input_shape):
        return [
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # rois
            (None, 1),  # class_ids
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # deltas
            (None, self.config.TRAIN_ROIS_PER_IMAGE, self.config.MASK_SHAPE[0],
             self.config.MASK_SHAPE[1])  # masks
        ]

    def compute_mask(self, inputs, mask=None):
        return [None, None, None, None]


############################################################
#  Detection Layer
############################################################

def clip_to_window(window, boxes):
    """
    window: (y1, x1, y2, x2). The window in the image we want to clip to.
    boxes: [N, (y1, x1, y2, x2)]
    """
    boxes[:, 0] = np.maximum(np.minimum(boxes[:, 0], window[2]), window[0])
    boxes[:, 1] = np.maximum(np.minimum(boxes[:, 1], window[3]), window[1])
    boxes[:, 2] = np.maximum(np.minimum(boxes[:, 2], window[2]), window[0])
    boxes[:, 3] = np.maximum(np.minimum(boxes[:, 3], window[3]), window[1])
    return boxes


def refine_detections(rois, probs, deltas, window, config):
    """Refine classified proposals and filter overlaps and return final
    detections.

    Inputs:
        rois: [N, (y1, x1, y2, x2)] in normalized coordinates
        probs: [N, num_classes]. Class probabilities.
        deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
                bounding box deltas.
        window: (y1, x1, y2, x2) in image coordinates. The part of the image
            that contains the image excluding the padding.

    Returns detections shaped: [N, (y1, x1, y2, x2, class_id, score)]
    """
    # Class IDs per ROI
    class_ids = np.argmax(probs, axis=1)
    # Class probability of the top class of each ROI
    class_scores = probs[np.arange(class_ids.shape[0]), class_ids]
    # Class-specific bounding box deltas
    deltas_specific = deltas[np.arange(deltas.shape[0]), class_ids]
    # Apply bounding box deltas
    # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    refined_rois = util.apply_box_deltas(
        rois, deltas_specific * config.BBOX_STD_DEV)
    # Convert coordiates to image domain
    # TODO: better to keep them normalized until later
    height, width = config.IMAGE_SHAPE[:2]
    refined_rois *= np.array([height, width, height, width])
    # Clip boxes to image window
    refined_rois = clip_to_window(window, refined_rois)
    # Round and cast to int since we're deadling with pixels now
    refined_rois = np.rint(refined_rois).astype(np.int32)

    # TODO: Filter out boxes with zero area

    # Filter out background boxes
    keep = np.where(class_ids > 0)[0]
    # Filter out low confidence boxes
    if config.DETECTION_MIN_CONFIDENCE:
        keep = np.intersect1d(
            keep, np.where(class_scores >= config.DETECTION_MIN_CONFIDENCE)[0])

    # Apply per-class NMS
    pre_nms_class_ids = class_ids[keep]
    pre_nms_scores = class_scores[keep]
    pre_nms_rois = refined_rois[keep]
    nms_keep = []
    for class_id in np.unique(pre_nms_class_ids):
        # Pick detections of this class
        ixs = np.where(pre_nms_class_ids == class_id)[0]
        # Apply NMS
        class_keep = util.non_max_suppression(
            pre_nms_rois[ixs], pre_nms_scores[ixs],
            config.DETECTION_NMS_THRESHOLD)
        # Map indicies
        class_keep = keep[ixs[class_keep]]
        nms_keep = np.union1d(nms_keep, class_keep)
    keep = np.intersect1d(keep, nms_keep).astype(np.int32)

    # Keep top detections
    roi_count = config.DETECTION_MAX_INSTANCES
    top_ids = np.argsort(class_scores[keep])[::-1][:roi_count]
    keep = keep[top_ids]

    # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
    # Coordinates are in image domain.
    result = np.hstack((refined_rois[keep],
                        class_ids[keep][..., np.newaxis],
                        class_scores[keep][..., np.newaxis]))
    return result


class DetectionLayer(KE.Layer):
    """Takes classified proposal boxes and their bounding box deltas and
    returns the final detection boxes.

    Returns:
    [batch, num_detections, (y1, x1, y2, x2, class_score)] in pixels
    """

    def __init__(self, config=None, **kwargs):
        super(DetectionLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        def wrapper(rois, mrcnn_class, mrcnn_bbox, image_meta):
            detections_batch = []
            for b in range(self.config.BATCH_SIZE):
                _, _, window, _ = parse_image_meta(image_meta)
                detections = refine_detections(
                    rois[b], mrcnn_class[b], mrcnn_bbox[b], window[b], self.config)
                # Pad with zeros if detections < DETECTION_MAX_INSTANCES
                gap = self.config.DETECTION_MAX_INSTANCES - detections.shape[0]
                assert gap >= 0
                if gap > 0:
                    detections = np.pad(
                        detections, [(0, gap), (0, 0)], 'constant', constant_values=0)
                detections_batch.append(detections)

            # Stack detections and cast to float32
            # TODO: track where float64 is introduced
            detections_batch = np.array(detections_batch).astype(np.float32)
            # Reshape output
            # [batch, num_detections, (y1, x1, y2, x2, class_score)] in pixels
            return np.reshape(detections_batch, [self.config.BATCH_SIZE, self.config.DETECTION_MAX_INSTANCES, 6])

        # Return wrapped function
        return tf.py_func(wrapper, inputs, tf.float32)

    def compute_output_shape(self, input_shape):
        return (None, self.config.DETECTION_MAX_INSTANCES, 6)


# Region Proposal Network (RPN)

def rpn_graph(feature_map, anchors_per_location, anchor_stride):
    """Builds the computation graph of Region Proposal Network.

    feature_map: backbone features [batch, feature_height=128, feature_width=128, feature_depth=256]
    anchors_per_location: number of anchors per pixel in the feature map, default 15=3*5
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).

    Returns:
        rpn_class_logits: [batch, 128*128*anchors_per_location(15=3*5), 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, 128*128*anchors_per_location(15=3*5), 2] Anchor classifier probabilities.
        rpn_bbox: [batch, H, W, (dy, dx, log(dh), log(dw))] Deltas to be
                  applied to anchors.
    """
    # TODO: check if stride of 2 causes alignment issues if the featuremap
    #       is not even.
    # Shared convolutional base of the RPN
    shared = KL.Conv2D(512, (3, 3), padding='same', activation='relu',
                       strides=anchor_stride,
                       name='rpn_conv_shared')(feature_map)

    # Anchor Score. [batch, height, width, anchors per location * 2].
    x = KL.Conv2D(2 * anchors_per_location, (1, 1), padding='valid',
                  activation='linear', name='rpn_class_raw')(shared)

    # Reshape to [batch, anchors, 2] for vip shape is [bacth, 128*128*anchors_per_location, 2] anchors_per_location=15
    rpn_class_logits = KL.Lambda(
        lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)

    # Softmax on last dimension of BG/FG.
    rpn_probs = KL.Activation(
        "softmax", name="rpn_class_xxx")(rpn_class_logits)

    # Bounding box refinement. [batch, H, W, anchors per location, depth]
    # where depth is [x, y, log(w), log(h)]
    x = KL.Conv2D(anchors_per_location * 4, (1, 1), padding="valid",
                  activation='linear', name='rpn_bbox_pred')(shared)

    # Reshape to [batch, 128*128*anchors_per_location(15=3*5), 4]
    rpn_bbox = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)

    return [rpn_class_logits, rpn_probs, rpn_bbox]


def build_rpn_model(anchor_stride, anchors_per_location, depth):
    """Builds a Keras model of the Region Proposal Network.
    It wraps the RPN graph so it can be used multiple times with shared
    weights.

    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).
    depth: Depth of the backbone feature map.

    Returns a Keras Model object. The model outputs, when called, are:
    rpn_logits: [batch, H, W, 2] Anchor classifier logits (before softmax)
    rpn_probs: [batch, W, W, 2] Anchor classifier probabilities.
    rpn_bbox: [batch, H, W, (dy, dx, log(dh), log(dw))] Deltas to be
                applied to anchors.
    """
    input_feature_map = KL.Input(shape=[None, None, depth],
                                 name="input_rpn_feature_map")
    outputs = rpn_graph(input_feature_map, anchors_per_location, anchor_stride)
    return KM.Model([input_feature_map], outputs, name="rpn_model")


############################################################
#  Feature Pyramid Network Heads
############################################################

def fpn_classifier_graph(rois, feature_map,
                         image_shape, pool_size, num_classes):
    """Builds the computation graph of the feature pyramid network classifier
    and regressor heads.

    rois: [batch, num_rois(default TRAIN_ROIS_PER_IMAGE=128, inference 1000),
        (y1, x1, y2, x2)] Proposal boxes in normalized coordinates to 1.
    feature_maps: [batch, f_height=128, f_width=128, f_chanel=256]
    image_shape: [height=512, width=512, depth=3]
    pool_size: The width of the square feature map generated from ROI Pooling. for example 7 in vip
    num_classes: number of classes, which determines the depth of the results. for example 2 in vip

    Returns:
        mrcnn_class_logits: [batch, num_boxes=128, num_classes] classifier logits (before softmax)
        mrcnn_probs: [batch, num_boxes=128, num_classes] classifier probabilities
        mrcnn_bbox: [batch, boxes, num_classes, (dy, dx, log(dh), log(dw))] Deltas to apply to
                     proposal boxes
    """
    # ROI Pooling
    # Shape: [batch, num_boxes=128, pool_height, pool_width, channels]
    x = ROIAlign([pool_size, pool_size], image_shape,
                 name="roi_align_classifier")([rois, feature_map])  # shape [1, batch * num_boxes, 7, 7, 256]

    # Two 1024 FC layers (implemented with Conv2D for consistency)
    x = KL.TimeDistributed(KL.Conv2D(1024, (pool_size, pool_size), padding="valid"),
                           name="mrcnn_class_conv1")(x)  # [batch, num_boxes=128, 1, 1, 1024]
    x = KL.Activation('relu')(x)
    x = KL.TimeDistributed(KL.Conv2D(1024, (1, 1)),
                           name="mrcnn_class_conv2")(x)  # [batch, num_boxes=128, 1, 1, 1024]
    x = KL.Activation('relu')(x)

    shared = KL.Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2),
                       name="pool_squeeze")(x)  # [batch, num_boxes=128, 1024]

    # Classifier head
    mrcnn_class_logits = KL.TimeDistributed(KL.Dense(num_classes), name='mrcnn_class_pascal_logits')(
        shared)  # [batch, num_boxes=128, num_classes]
    mrcnn_probs = KL.TimeDistributed(KL.Activation("softmax"), name="mrcnn_class")(
        mrcnn_class_logits)  # [batch, num_boxes=128, num_classes]

    # BBox head
    # [batch, boxes, num_classes * (dy, dx, log(dh), log(dw))]
    x = KL.TimeDistributed(KL.Dense(num_classes * 4, activation='linear'),
                           name='mrcnn_bbox_pascal_fc')(shared)  # shape [batch, num_boxes=128, num_classes * 4]
    # Reshape to [batch, boxes, num_classes, (dy, dx, log(dh), log(dw))]
    s = K.int_shape(x)
    mrcnn_bbox = KL.Reshape((s[1], num_classes, 4), name="mrcnn_bbox")(x)  # [batch, num_boxes=128, num_classes, 4]

    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox


def build_fpn_mask_graph(rois, feature_map,
                         image_shape, pool_size, num_classes):
    """Builds the computation graph of the mask head of Feature Pyramid Network.

    rois: [batch, num_boxes, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.num_boxes training 128, inference 100
    feature_maps:
    image_shape: [height, width, depth]
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results

    Returns: Masks [batch, roi_count, height, width, num_classes]
    """
    # ROI Pooling
    # Shape: [batch_size, num_boxes=TRAIN_ROIS_PER_IMAGE=128, pool_height, pool_width, channels]
    x = ROIAlign([pool_size, pool_size], image_shape,
                 name="roi_align_mask")([rois, feature_map])  # shape [1, batch * num_boxes, 14, 14, 256]

    # Conv layers
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv1")(x)  # shape [batch, num_boxes=128, 14, 14, 256]
    x = KL.TimeDistributed(BatchNorm(axis=3),
                           name='mrcnn_mask_bn1')(x)  # shape [batch, num_boxes, 14, 14, 256]
    conv1 = KL.Activation('relu')(x)  # shape [batch, num_boxes, 14, 14, 256]

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv2")(conv1)
    x = KL.TimeDistributed(BatchNorm(axis=3),
                           name='mrcnn_mask_bn2')(x)
    conv2 = KL.Activation('relu')(x)  # shape [batch, num_boxes, 14, 14, 256]

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv3")(conv2)
    x = KL.TimeDistributed(BatchNorm(axis=3),
                           name='mrcnn_mask_bn3')(x)
    conv3 = KL.Activation('relu')(x)  # shape [batch, num_boxes, 14, 14, 256]

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv4")(conv3)
    x = KL.TimeDistributed(BatchNorm(axis=3),
                           name='mrcnn_mask_bn4')(x)
    conv4 = KL.Activation('relu')(x)  # shape [batch, num_boxes, 14, 14, 256]

    x = KL.TimeDistributed(KL.Conv2DTranspose(256, (2, 2), strides=2, activation="relu"),
                           name="mrcnn_mask_deconv")(conv4)  # shape [batch, num_boxes, 28, 28, 256]
    x = KL.TimeDistributed(KL.Conv2D(num_classes, (1, 1), strides=1, activation="sigmoid"),
                           name="mrcnn_mask_pascal")(x)  # shape [batch, num_boxes, 28, 28, 2]

    conv4_fc = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                                  name="mrcnn_mask_conv4_fc")(conv3)  # shape [batch, num_boxes, 14, 14, 256]
    conv4_fc = KL.TimeDistributed(BatchNorm(axis=3),
                                  name='mrcnn_mask_bn4_fc')(conv4_fc)  # shape [batch, num_boxes, 14, 14, 256]
    conv4_fc = KL.Activation('relu')(conv4_fc)  # shape [batch, num_boxes, 14, 14, 256]
    conv5_fc = KL.TimeDistributed(KL.Conv2D(128, (3, 3), padding="same"),
                                  name="mrcnn_mask_conv5_fc")(conv4_fc)  # shape [batch, num_boxes, 14, 14, 256]
    conv5_fc = KL.TimeDistributed(BatchNorm(axis=3),
                                  name='mrcnn_mask_bn5_fc')(conv5_fc)  # shape [batch, num_boxes, 14, 14, 256]
    conv5_fc = KL.Activation('relu')(conv5_fc)  # shape [batch, num_boxes, 14, 14, 256]

    fc = KL.TimeDistributed(KL.Conv2D(784 * num_classes, (pool_size, pool_size), name="mrcnn_mask_fc"))(conv5_fc)
    # fc: shape [batch, 128, 1, 1, 1568=784(28*28) * num_classes], x: shape [batch, num_boxes, 28, 28, 2]
    x1 = KL.Lambda(lambda x: tf.reshape(x[0], tf.shape(x[1])))([fc, x])
    # x1 shape [batch, 128, 28, 28, 2]
    final_mask = KL.Add()([x, x1])  # shape [batch, 128, 28, 28, 2]
    return final_mask


def arbitrary_size_pooling(feature_map):
    """similar from  keras.layers import GlobalAveragePooling2D

    Args:
        feature_map:  Tensor("activation_141/Relu:0", shape=(?, 32, 32, 2048), dtype=float32)

    Returns:
        b1: Tensor("lambda_1/Mean:0", shape=(?, 1, 32, 2048), dtype=float32)
        b2: Tensor("lambda_1/Mean_3:0", shape=(?, 1, 1, 2048), dtype=float32)
    """
    b1 = tf.reduce_mean(feature_map, axis=1, keep_dims=True)
    b2 = tf.reduce_mean(b1, axis=2, keep_dims=True)
    return b2


def global_parsing_encoder(feature_map):
    x1 = KL.Conv2D(256, (1, 1), padding='same',
                   name='mrcnn_global_parsing_encoder_c1')(feature_map)
    # x1 = BatchNorm(axis=-1, name='mrcnn_global_parsing_encoder_bn1')(x1)
    x1 = KL.Activation('relu')(x1)

    x2 = KL.Conv2D(256, (3, 3), padding='same', dilation_rate=(6, 6),
                   name='mrcnn_global_parsing_encoder_c2')(feature_map)
    # x2 = BatchNorm(axis=-1, name='mrcnn_global_parsing_encoder_bn2')(x2)
    x2 = KL.Activation('relu')(x2)

    x3 = KL.Conv2D(256, (3, 3), padding='same', dilation_rate=(12, 12),
                   name='mrcnn_global_parsing_encoder_c3')(feature_map)
    # x3 = BatchNorm(axis=-1, name='mrcnn_global_parsing_encoder_bn3')(x3)
    x3 = KL.Activation('relu')(x3)

    x4 = KL.Conv2D(256, (3, 3), padding='same', dilation_rate=(18, 18),
                   name='mrcnn_global_parsing_encoder_c4')(feature_map)
    # x4 = BatchNorm(axis=-1, name='mrcnn_global_parsing_encoder_bn4')(x4)
    x4 = KL.Activation('relu')(x4)

    x0 = KL.Lambda(lambda x: arbitrary_size_pooling(x))(feature_map)
    x0 = KL.Conv2D(256, (1, 1), padding='same',
                   name='mrcnn_global_parsing_encoder_c0')(x0)
    # x0 = BatchNorm(axis=-1, name='mrcnn_global_parsing_encoder_bn0')(x0)
    x0 = KL.Activation('relu')(x0)
    x0 = KL.Lambda(lambda x: tf.image.resize_bilinear(
        x[0], tf.shape(x[1])[1:3], align_corners=True))([x0, feature_map])

    x = KL.Lambda(lambda x: tf.concat(x, axis=-1))([x0, x1, x2, x3, x4])
    x = KL.Conv2D(256, (1, 1), padding='same',
                  name='mrcnn_global_parsing_encoder_conconv')(x)
    # x = BatchNorm(axis=-1, name='mrcnn_global_parsing_encoder_bn')(x)
    x = KL.Activation('relu')(x)

    return x


def global_parsing_decoder(feature_map, low_feature_map):
    # navie upsample from 1/16(32) to 1/4(128), fit the low_feature_map
    top = KL.Lambda(lambda x: tf.image.resize_bilinear(
        x[0], tf.shape(x[1])[1:3], align_corners=True))([feature_map, low_feature_map])
    # low dim of low_feature_map by 1*1 conv
    low = KL.Conv2D(48, (1, 1), padding='same',
                    name='mrcnn_global_parsing_decoder_conv1')(low_feature_map)
    low = KL.Activation('relu')(low)

    # x = KL.Concatenate(axis=-1)([top, low])
    x = KL.Lambda(lambda x: tf.concat(x, axis=-1))([top, low])
    x = KL.Conv2D(256, (3, 3), padding='same',
                  name='mrcnn_global_parsing_decoder_conv2')(x)
    x = KL.Activation('relu')(x)
    x = KL.Conv2D(256, (3, 3), padding='same',
                  name='mrcnn_global_parsing_decoder_conv3')(x)
    x = KL.Activation('relu')(x)
    return x


def global_parsing_graph(feature_map, num_classes):
    x1 = KL.Conv2D(num_classes, (3, 3), padding='same', dilation_rate=(6, 6),
                   name='mrcnn_global_parsing_c1')(feature_map)

    x2 = KL.Conv2D(num_classes, (3, 3), padding='same', dilation_rate=(12, 12),
                   name='mrcnn_global_parsing_c2')(feature_map)

    x3 = KL.Conv2D(num_classes, (3, 3), padding='same', dilation_rate=(18, 18),
                   name='mrcnn_global_parsing_c3')(feature_map)

    x = KL.Add()([x1, x2, x3])
    return x


############################################################
#  Loss Functions
############################################################

def smooth_l1_loss(y_true, y_pred):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typicallly: [N, 4], but could be any shape.
    """
    diff = K.abs(y_true - y_pred)  # [N, 4]
    less_than_one = K.cast(K.less(diff, 1.0), "float32")  # [N, 4]
    loss = (less_than_one * 0.5 * diff ** 2) + (1 - less_than_one) * (diff - 0.5)  # [N, 4]
    return loss


def rpn_class_loss_graph(rpn_match, rpn_class_logits):
    """RPN anchor classifier loss.

    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for FG/BG.
    """
    # Squeeze last dim to simplify
    rpn_match = tf.squeeze(rpn_match, -1)  # [batch, anchors]
    # Get anchor classes. Convert the -1/+1 match to 0/1 values.
    anchor_class = K.cast(K.equal(rpn_match, 1), tf.int32)  # [batch, anchors]
    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    indices = tf.where(K.not_equal(rpn_match, 0))  # [num_not0, 2]
    # Pick rows that contribute to the loss and filter out the rest.
    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)  # [num_not0, 2]
    anchor_class = tf.gather_nd(anchor_class, indices)  # [num_not0, ]
    # Crossentropy loss
    loss = K.sparse_categorical_crossentropy(target=anchor_class,
                                             output=rpn_class_logits,
                                             from_logits=True)
    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    return K.cast(loss, dtype="float32")


def rpn_bbox_loss_graph(config, target_bbox, rpn_match, rpn_bbox):
    """Return the RPN bounding box loss graph.

    config: the model config object.
    target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
        Uses 0 padding to fill in unsed bbox deltas.
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
    """
    # Positive anchors contribute to the loss, but negative and
    # neutral anchors (match value of 0 or -1) don't.
    rpn_match = K.squeeze(rpn_match, -1)
    indices = tf.where(K.equal(rpn_match, 1))

    # Pick bbox deltas that contribute to the loss
    rpn_bbox = tf.gather_nd(rpn_bbox, indices)  # [num_eq1, 4]

    # Trim target bounding box deltas to the same length as rpn_bbox.
    batch_counts = K.sum(K.cast(K.equal(rpn_match, 1), tf.int32), axis=1)  # [batch, ]
    target_bbox = batch_pack_graph(target_bbox, batch_counts,
                                   config.IMAGES_PER_GPU)  # [num_eq1, 4]

    # TODO: use smooth_l1_loss() rather than reimplementing here
    #       to reduce code duplication
    diff = K.abs(target_bbox - rpn_bbox)  # [num_eq1, 4]
    less_than_one = K.cast(K.less(diff, 1.0), "float32")  # [num_eq1, 4]
    loss = (less_than_one * 0.5 * diff ** 2) + (1 - less_than_one) * (diff - 0.5)  # [num_eq1, 4]

    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))  # [1,]
    return K.cast(loss, dtype="float32")


def mrcnn_class_loss_graph(target_class_ids, pred_class_logits,
                           active_class_ids):
    """Loss for the classifier head of Mask RCNN.

    target_class_ids: [batch, num_rois=TRAIN_ROIS_PER_IMAGE=128]. Integer class IDs. Uses zero
        padding to fill in the array.
    pred_class_logits: [batch, num_rois, num_classes]
    active_class_ids: [batch, num_classes]. Has a value of 1 for
        classes that are in the dataset of the image, and 0
        for classes that are not in the dataset.
    """
    target_class_ids = tf.cast(target_class_ids, 'int64')  # [batch, num_rois]

    # Find predictions of classes that are not in the dataset.
    pred_class_ids = tf.argmax(pred_class_logits, axis=2)  # [batch, num_rois]
    # TODO: Update this line to work with batch > 1. Right now it assumes all
    #       images in a batch have the same active_class_ids
    pred_active = tf.gather(active_class_ids[0], pred_class_ids)  # [batch, num_rois]

    # Loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_class_ids, logits=pred_class_logits)  # [batch, num_rois]

    # Erase losses of predictions of classes that are not in the active
    # classes of the image.
    loss = loss * pred_active  # [batch, num_rois]

    # Computer loss mean. Use only predictions that contribute
    # to the loss to get a correct mean.
    loss = tf.reduce_sum(loss) / tf.reduce_sum(pred_active)
    return K.cast(loss, dtype="float32")


def mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox):
    """Loss for Mask R-CNN bounding box refinement.

    target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    """
    # Reshape to merge batch and roi dimensions for simplicity.
    target_class_ids = K.reshape(target_class_ids, (-1,))  # [batch * num_rois, ]
    target_bbox = K.reshape(target_bbox, (-1, 4))  # [batch * num_rois, (dy, dx, log(dh), log(dw))]
    pred_bbox = K.reshape(pred_bbox, (-1, K.int_shape(pred_bbox)[2], 4))  # [batch * num_rois, num_classes, 4]]

    # Only positive ROIs contribute to the loss. And only
    # the right class_id of each ROI. Get their indicies.
    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]  # [num_great0, ]
    positive_roi_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_roi_ix), tf.int64)  # [num_great0, ]
    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)  # [num_great0, 2]

    # Gather the deltas (predicted and true) that contribute to loss
    target_bbox = tf.gather(target_bbox, positive_roi_ix)  # [num_great0, 4]
    pred_bbox = tf.gather_nd(pred_bbox, indices)  # [num_great0, 4]

    # Smooth-L1 Loss
    loss = K.switch(tf.size(target_bbox) > 0,
                    smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox),
                    tf.constant(0.0))
    loss = K.mean(loss)  # [1, ]
    # loss = K.reshape(loss, [1, 1])
    return K.cast(loss, dtype="float32")


def mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks):
    """Mask binary cross-entropy loss for the masks head.

    target_masks: [batch, num_rois, m_height=28, m_width=28].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                with values from 0 to 1.
    """
    # Reshape for simplicity. Merge first two dimensions into one.
    target_class_ids = K.reshape(target_class_ids, (-1,))  # [batch * num_rois, ]
    mask_shape = tf.shape(target_masks)
    target_masks = K.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))  # [batch * num_rois, 28, 28]
    pred_shape = tf.shape(pred_masks)
    pred_masks = K.reshape(pred_masks,
                           (-1, pred_shape[2], pred_shape[3], pred_shape[4]))  # [batch * num_rois, 28, 28, 2]
    # Permute predicted masks to [N, num_classes, height, width]
    pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])  # [batch * num_rois, 2, 28, 28]

    # Only positive ROIs contribute to the loss. And only
    # the class specific mask of each ROI.
    positive_ix = tf.where(target_class_ids > 0)[:, 0]  # [num_great0, ]
    positive_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_ix), tf.int64)  # [num_great0, ]
    indices = tf.stack([positive_ix, positive_class_ids], axis=1)  # [num_great0, 2]

    # Gather the masks (predicted and true) that contribute to loss
    y_true = tf.gather(target_masks, positive_ix)  # [num_great0, 28, 28]
    y_pred = tf.gather_nd(pred_masks, indices)  # [num_great0, 28, 28]

    # Compute binary cross entropy. If no positive ROIs, then return 0.
    # shape: [batch, roi, num_classes]
    loss = K.switch(tf.size(y_true) > 0,
                    K.binary_crossentropy(target=y_true, output=y_pred),
                    tf.constant(0.0))  # [num_great0, 28, 28]
    loss = K.mean(loss)
    # loss = K.reshape(loss, [1, 1])
    return K.cast(loss, dtype="float32")


def mrcnn_global_parsing_loss_graph(num_classes, gt_parsing_map, predict_parsing_map):
    """
    num_classes: for part parsing is 20= 1 + 19 (background + classes)
    gt_parsing_map: [batch, image_height=512, image_width=512] of uint8
    predict_parsing_map: [batch, f_height=128, f_width=128, num_classes=20]
    """
    gt_shape = tf.shape(gt_parsing_map)  # the value is [batch, 512, 512]
    predict_parsing_map = tf.image.resize_bilinear(predict_parsing_map, gt_shape[1:3])  # shape [batch, 512, 512, 20]

    pred_shape = tf.shape(predict_parsing_map)  # the value is [batch, 512, 512, 20]

    raw_gt = tf.expand_dims(gt_parsing_map, -1)  # shape [batch, 512, 512, 1]
    # raw_gt = tf.image.resize_nearest_neighbor(raw_gt, pred_shape[1:3])
    raw_gt = tf.reshape(raw_gt, [-1, ])  # shape [batch*512*512*1, ]
    raw_gt = tf.cast(raw_gt, tf.int32)

    raw_prediction = tf.reshape(predict_parsing_map, [-1, pred_shape[-1]])  # shape [batch*512*512, 20]

    # Predictions: ignoring all predictions with labels greater or equal than n_classes
    indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, num_classes - 1)), 1)  # shape [K=(batch*512*512 - ignore_num),]
    gt = tf.gather(raw_gt, indices)  # shape = (K,) K 表示找到的符合tf.less_equal(raw_gt, num_classes - 1)的个数
    prediction = tf.gather(raw_prediction, indices)  # shape = （K, num_classes=20)
    # gt: shape [K,], the value is [0,19], 0 is bg, other represent the part lable,
    # prediction: shape[K, parsing_class_num=20],
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=gt, logits=prediction)  # (num_classes, )
    loss = tf.reduce_mean(loss)  # scalar
    # loss = tf.reshape(loss, [1, 1])
    return K.cast(loss, dtype="float32")


def post_processing_graph(parts, input_image):
    parts = tf.image.resize_bilinear(parts, tf.shape(input_image)[1:3, ])
    parts = tf.nn.softmax(parts)
    return parts


def load_image_gt(dataset, config, image_id, augment=False,
                  use_mini_mask=False):
    """Load and return ground truth data for an image (image, mask, bounding boxes).
    Args:
        augment: If true, apply random image augmentation. Currently, only
            horizontal flipping is offered.
        use_mini_mask: If False, returns full-size masks that are the same height
            and width as the original image. These can be big, for example
            1024x1024x100 (for 100 instances). Mini masks are smaller, typically,
            224x224 and are generated by extracting the bounding box of the
            object and resizing it to MINI_MASK_SHAPE.

    Returns:
        image: [height(IMAGE_MAX_DIM=512), width(IMAGE_MAX_DIM=512), 3]
        image_meta:
        class_ids: [instance_count] Integer class IDs
        bbox: [instance_count, (y1, x1, y2, x2)]
        mask: [height(default 56), width(default 56), instance_count]. The height and width are those
            of the image unless use_mini_mask is True, in which case they are
            defined in MINI_MASK_SHAPE=56.
        part: [resize_height(IMAGE_MAX_DIM=512), resize_width(IMAGE_MAX_DIM=512)]
            the value is 0-19, 0 is bg, 1-19 is the person part label
    """
    # Load image and mask
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    part = dataset.load_part(image_id)
    part_rev = dataset.load_reverse_part(image_id)
    shape = image.shape

    image, window, scale, padding = util.resize_image(
        image,
        max_dim=config.IMAGE_MAX_DIM,
        padding=config.IMAGE_PADDING)
    mask = util.resize_mask(mask, scale, padding)
    part = util.resize_part(part, scale, padding[:2])
    part_rev = util.resize_part(part_rev, scale, padding[:2])
    # Random horizontal flips.
    if augment:
        if random.randint(0, 1):
            image = np.fliplr(image)
            mask = np.fliplr(mask)
            # part = np.fliplr(part)
            part = part_rev

    # Bounding boxes. Note that some boxes might be all zeros
    # if the corresponding mask got cropped out.
    # bbox: [num_instances, (y1, x1, y2, x2)]
    bbox = util.extract_bboxes(mask)

    # Active classes
    # Different datasets have different classes, so track the
    # classes supported in the dataset of this image.
    active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
    source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
    active_class_ids[source_class_ids] = 1

    # Resize masks to smaller size to reduce memory usage
    if use_mini_mask:
        mask = util.minimize_mask(bbox, mask, config.MINI_MASK_SHAPE)

    # Image meta data
    image_meta = compose_image_meta(image_id, shape, window, active_class_ids)
    return image, image_meta, class_ids, bbox, mask, part, scale


def build_rpn_targets(image_shape, anchors, gt_class_ids, gt_boxes, config):
    """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.
    Args:
        anchors: [num_anchors=245760=128*128*3*5, (y1, x1, y2, x2)]
        gt_class_ids: [num_gt_boxes] Integer class IDs.
        gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]

    Returns:
        rpn_match: [num_anchors=245760=128*128*3*5, ] (int32) matches between anchors and GT boxes.
                   1 = positive anchor, -1 = negative anchor, 0 = neutral,
                   num_positive_anchor + num_negative_anchor = config.RPN_TRAIN_ANCHORS_PER_IMAGE = 256
        rpn_bbox: [config.RPN_TRAIN_ANCHORS_PER_IMAGE=256, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    """
    # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)  # shape [num_anchors=245760=128*128*3*5, ] (int32)
    # RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
    rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))  # shape[config.RPN_TRAIN_ANCHORS_PER_IMAGE=256, 4]

    # Compute overlaps [num_anchors, num_gt_boxes]
    overlaps = util.compute_overlaps(anchors, gt_boxes)

    # Match anchors to GT Boxes
    # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
    # If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
    # Neutral anchors are those that don't match the conditions above,
    # and they don't influence the loss function.
    # However, don't keep any GT box unmatched (rare, but happens). Instead,
    # match it to the closest anchor (even if its max IoU is < 0.3).
    #
    # 1. Set negative anchors first. They get overwritten below if a GT box is
    # matched to them. Skip boxes in crowd areas.
    anchor_iou_argmax = np.argmax(overlaps, axis=1)  # shape : [num_anchors=245760, ]
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]  # shape : [num_anchors=245760, ]
    rpn_match[(anchor_iou_max < 0.3)] = -1
    # 2. Set an anchor for each GT box (regardless of IoU value).
    # TODO: If multiple anchors have the same IoU match all of them
    gt_iou_argmax = np.argmax(overlaps, axis=0)  # shape [num_gt_boxes, ]
    rpn_match[gt_iou_argmax] = 1
    # 3. Set anchors with high overlap as positive.
    rpn_match[anchor_iou_max >= 0.7] = 1

    # Subsample to balance positive and negative anchors
    # Don't let positives be more than half the anchors
    ids = np.where(rpn_match == 1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
    if extra > 0:
        # Reset the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    # Same for negative proposals
    ids = np.where(rpn_match == -1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE -
                        np.sum(rpn_match == 1))
    if extra > 0:
        # Rest the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0

    # For positive anchors, compute shift and scale needed to transform them
    # to match the corresponding GT boxes.
    ids = np.where(rpn_match == 1)[0]  # For example: ids shape: (41,)
    ix = 0  # index into rpn_bbox
    # TODO: use box_refinment() rather than duplicating the code here
    for i, a in zip(ids, anchors[ids]):
        # Closest gt box (it might have IoU < 0.7)
        gt = gt_boxes[anchor_iou_argmax[i]]

        # Convert coordinates to center plus width/height.
        # GT Box
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w
        # Anchor
        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w

        # Compute the bbox refinement that the RPN should predict.  rpn_bbox shape: (256, 4)
        rpn_bbox[ix] = [
            (gt_center_y - a_center_y) / a_h,
            (gt_center_x - a_center_x) / a_w,
            np.log(gt_h / a_h),
            np.log(gt_w / a_w),
        ]
        # Normalize
        rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV  # config.RPN_BBOX_STD_DEV ndarray [0.1 0.1 0.2 0.2]
        ix += 1

    return rpn_match, rpn_bbox


def build_detection_targets(rpn_rois, gt_class_ids, gt_boxes, gt_masks, gt_parts, config):
    """Generate targets for training Stage 2 classifier and mask heads.
    This is not used in normal training. It's useful for debugging or to train
    the Mask RCNN heads without using the RPN head.
    Args:
        rpn_rois: [N, (y1, x1, y2, x2)] proposal boxes.
        gt_class_ids: [instance count] Integer class IDs
        gt_boxes: [instance count, (y1, x1, y2, x2)]
        gt_masks: [height(default MINI_MASK_SHAPE=56), width, instance count] Ground truth masks. Can be full
                  size or mini-masks.
        gt_parts: [resize_height(IMAGE_MAX_DIM=512), resize_width(IMAGE_MAX_DIM=512)],
            the value is 0-19, 0 is bg, 1-19 is the person part label
    Returns:
        rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)]
        class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
        bboxes: [TRAIN_ROIS_PER_IMAGE, NUM_CLASSES, (y, x, log(h), log(w))]. Class-specific
                bbox refinements.
        masks: [TRAIN_ROIS_PER_IMAGE, height, width, NUM_CLASSES). Class specific masks cropped
               to bbox boundaries and resized to neural network output size.
        parts:
    """
    assert rpn_rois.shape[0] > 0
    assert gt_class_ids.dtype == np.int32, "Expected int but got {}".format(
        gt_class_ids.dtype)
    assert gt_boxes.dtype == np.int32, "Expected int but got {}".format(
        gt_boxes.dtype)
    assert gt_masks.dtype == np.bool_, "Expected bool but got {}".format(
        gt_masks.dtype)
    assert gt_parts.dtype == np.uint8, "Expected bool but got {}".format(
        gt_parts.dtype)

    # It's common to add GT Boxes to ROIs but we don't do that here because
    # according to XinLei Chen's paper, it doesn't help.

    # Trim empty padding in gt_boxes and gt_masks parts
    instance_ids = np.where(gt_class_ids > 0)[0]
    assert instance_ids.shape[0] > 0, "Image must contain instances."
    gt_class_ids = gt_class_ids[instance_ids]
    gt_boxes = gt_boxes[instance_ids]
    gt_masks = gt_masks[:, :, instance_ids]

    # Compute areas of ROIs and ground truth boxes.
    rpn_roi_area = (rpn_rois[:, 2] - rpn_rois[:, 0]) * \
                   (rpn_rois[:, 3] - rpn_rois[:, 1])
    gt_box_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * \
                  (gt_boxes[:, 3] - gt_boxes[:, 1])

    # Compute overlaps [rpn_rois, gt_boxes]
    overlaps = np.zeros((rpn_rois.shape[0], gt_boxes.shape[0]))
    for i in range(overlaps.shape[1]):
        gt = gt_boxes[i]
        overlaps[:, i] = util.compute_iou(
            gt, rpn_rois, gt_box_area[i], rpn_roi_area)

    # Assign ROIs to GT boxes
    rpn_roi_iou_argmax = np.argmax(overlaps, axis=1)
    rpn_roi_iou_max = overlaps[np.arange(
        overlaps.shape[0]), rpn_roi_iou_argmax]
    # GT box assigned to each ROI
    rpn_roi_gt_boxes = gt_boxes[rpn_roi_iou_argmax]
    rpn_roi_gt_class_ids = gt_class_ids[rpn_roi_iou_argmax]

    # Positive ROIs are those with >= 0.5 IoU with a GT box.
    fg_ids = np.where(rpn_roi_iou_max > 0.5)[0]

    # Negative ROIs are those with max IoU 0.1-0.5 (hard example mining)
    # TODO: To hard example mine or not to hard example mine, that's the question
    # bg_ids = np.where((rpn_roi_iou_max >= 0.1) & (rpn_roi_iou_max < 0.5))[0]
    bg_ids = np.where(rpn_roi_iou_max < 0.5)[0]

    # Subsample ROIs. Aim for 33% foreground.
    # FG
    fg_roi_count = int(config.TRAIN_ROIS_PER_IMAGE * config.ROI_POSITIVE_RATIO)
    if fg_ids.shape[0] > fg_roi_count:
        keep_fg_ids = np.random.choice(fg_ids, fg_roi_count, replace=False)
    else:
        keep_fg_ids = fg_ids
    # BG
    remaining = config.TRAIN_ROIS_PER_IMAGE - keep_fg_ids.shape[0]
    if bg_ids.shape[0] > remaining:
        keep_bg_ids = np.random.choice(bg_ids, remaining, replace=False)
    else:
        keep_bg_ids = bg_ids
    # Combine indices of ROIs to keep
    keep = np.concatenate([keep_fg_ids, keep_bg_ids])
    # Need more?
    remaining = config.TRAIN_ROIS_PER_IMAGE - keep.shape[0]
    if remaining > 0:
        # Looks like we don't have enough samples to maintain the desired
        # balance. Reduce requirements and fill in the rest. This is
        # likely different from the Mask RCNN paper.

        # There is a small chance we have neither fg nor bg samples.
        if keep.shape[0] == 0:
            # Pick bg regions with easier IoU threshold
            bg_ids = np.where(rpn_roi_iou_max < 0.5)[0]
            assert bg_ids.shape[0] >= remaining
            keep_bg_ids = np.random.choice(bg_ids, remaining, replace=False)
            assert keep_bg_ids.shape[0] == remaining
            keep = np.concatenate([keep, keep_bg_ids])
        else:
            # Fill the rest with repeated bg rois.
            keep_extra_ids = np.random.choice(
                keep_bg_ids, remaining, replace=True)
            keep = np.concatenate([keep, keep_extra_ids])
    assert keep.shape[0] == config.TRAIN_ROIS_PER_IMAGE, \
        "keep doesn't match ROI batch size {}, {}".format(
            keep.shape[0], config.TRAIN_ROIS_PER_IMAGE)

    # Reset the gt boxes assigned to BG ROIs.
    rpn_roi_gt_boxes[keep_bg_ids, :] = 0
    rpn_roi_gt_class_ids[keep_bg_ids] = 0

    # For each kept ROI, assign a class_id, and for FG ROIs also add bbox refinement.
    rois = rpn_rois[keep]
    roi_gt_boxes = rpn_roi_gt_boxes[keep]
    roi_gt_class_ids = rpn_roi_gt_class_ids[keep]
    roi_gt_assignment = rpn_roi_iou_argmax[keep]

    # Class-aware bbox deltas. [y, x, log(h), log(w)]
    bboxes = np.zeros((config.TRAIN_ROIS_PER_IMAGE,
                       config.NUM_CLASSES, 4), dtype=np.float32)
    pos_ids = np.where(roi_gt_class_ids > 0)[0]
    bboxes[pos_ids, roi_gt_class_ids[pos_ids]] = util.box_refinement(
        rois[pos_ids], roi_gt_boxes[pos_ids, :4])
    # Normalize bbox refinements
    bboxes /= config.BBOX_STD_DEV

    # Generate class-specific target masks
    masks = np.zeros((config.TRAIN_ROIS_PER_IMAGE, config.MASK_SHAPE[0], config.MASK_SHAPE[1], config.NUM_CLASSES),
                     dtype=np.float32)
    for i in pos_ids:
        class_id = roi_gt_class_ids[i]
        assert class_id > 0, "class id must be greater than 0"
        gt_id = roi_gt_assignment[i]
        class_mask = gt_masks[:, :, gt_id]

        if config.USE_MINI_MASK:
            # Create a mask placeholder, the size of the image
            placeholder = np.zeros(config.IMAGE_SHAPE[:2], dtype=bool)
            # GT box
            gt_y1, gt_x1, gt_y2, gt_x2 = gt_boxes[gt_id]
            gt_w = gt_x2 - gt_x1
            gt_h = gt_y2 - gt_y1
            # Resize mini mask to size of GT box
            placeholder[gt_y1:gt_y2, gt_x1:gt_x2] = \
                np.round(skimage.transform.resize(class_mask, (gt_h, gt_w), order=1, mode="constant")).astype(bool)
            # Place the mini batch in the placeholder
            class_mask = placeholder

        # Pick part of the mask and resize it
        y1, x1, y2, x2 = rois[i].astype(np.int32)
        m = class_mask[y1:y2, x1:x2]
        mask = skimage.transform.resize(m, config.MASK_SHAPE, order=1, mode="constant")
        masks[i, :, :, class_id] = mask

    return rois, roi_gt_class_ids, bboxes, masks, gt_parts


def generate_random_rois(image_shape, count, gt_boxes):
    """Generates ROI proposals similar to what a region proposal network
    would generate.
    Args:
        image_shape: [resize_height=512, resize_width=512, Depth]
        count: Number of ROIs to generate
        gt_boxes: list, [N, (y1, x1, y2, x2)] Ground truth boxes in pixels. Needed N > 0
    Returns:
        rois: ndarray, [count, (y1, x1, y2, x2)] ROI boxes in pixels.
    """
    # placeholder
    rois = np.zeros((count, 4), dtype=np.int32)
    gt_boxes = np.array(gt_boxes)
    # Generate random ROIs around GT boxes (90% of count)
    rois_per_box = int(0.9 * count / gt_boxes.shape[0])
    for i in range(gt_boxes.shape[0]):
        gt_y1, gt_x1, gt_y2, gt_x2 = gt_boxes[i]
        h = gt_y2 - gt_y1
        w = gt_x2 - gt_x1
        # random boundaries
        r_y1 = max(gt_y1 - h, 0)
        r_y2 = min(gt_y2 + h, image_shape[0])
        r_x1 = max(gt_x1 - w, 0)
        r_x2 = min(gt_x2 + w, image_shape[1])

        # To avoid generating boxes with zero area, we generate double what
        # we need and filter out the extra. If we get fewer valid boxes
        # than we need, we loop and try again.
        while True:
            y1y2 = np.random.randint(r_y1, r_y2, (rois_per_box * 2, 2))
            x1x2 = np.random.randint(r_x1, r_x2, (rois_per_box * 2, 2))
            # Filter out zero area boxes
            threshold = 1
            y1y2 = y1y2[np.abs(y1y2[:, 0] - y1y2[:, 1]) >=
                        threshold][:rois_per_box]
            x1x2 = x1x2[np.abs(x1x2[:, 0] - x1x2[:, 1]) >=
                        threshold][:rois_per_box]
            if y1y2.shape[0] == rois_per_box and x1x2.shape[0] == rois_per_box:
                break

        # Sort on axis 1 to ensure x1 <= x2 and y1 <= y2 and then reshape
        # into x1, y1, x2, y2 order
        x1, x2 = np.split(np.sort(x1x2, axis=1), 2, axis=1)
        y1, y2 = np.split(np.sort(y1y2, axis=1), 2, axis=1)
        box_rois = np.hstack([y1, x1, y2, x2])
        rois[rois_per_box * i:rois_per_box * (i + 1)] = box_rois

    # Generate random ROIs anywhere in the image (10% of count)
    remaining_count = count - (rois_per_box * gt_boxes.shape[0])
    # To avoid generating boxes with zero area, we generate double what
    # we need and filter out the extra. If we get fewer valid boxes
    # than we need, we loop and try again.
    while True:
        y1y2 = np.random.randint(0, image_shape[0], (remaining_count * 2, 2))
        x1x2 = np.random.randint(0, image_shape[1], (remaining_count * 2, 2))
        # Filter out zero area boxes
        threshold = 1
        y1y2 = y1y2[np.abs(y1y2[:, 0] - y1y2[:, 1]) >=
                    threshold][:remaining_count]
        x1x2 = x1x2[np.abs(x1x2[:, 0] - x1x2[:, 1]) >=
                    threshold][:remaining_count]
        if y1y2.shape[0] == remaining_count and x1x2.shape[0] == remaining_count:
            break

    # Sort on axis 1 to ensure x1 <= x2 and y1 <= y2 and then reshape
    # into x1, y1, x2, y2 order
    x1, x2 = np.split(np.sort(x1x2, axis=1), 2, axis=1)
    y1, y2 = np.split(np.sort(y1y2, axis=1), 2, axis=1)
    global_rois = np.hstack([y1, x1, y2, x2])
    rois[-remaining_count:] = global_rois
    return rois


def generate_rois(image_shape, counts, gt_boxes):
    """Generates ROI proposals similar to what a region proposal network
    would generate.
    Args:
        image_shape: [resize_height=512, resize_width=512, Depth]
        counts: Number of ROIs to generate
        gt_boxes: list, [N, (y1, x1, y2, x2)] Ground truth boxes in pixels. Needed N > 0
    Returns:
        rois: ndarray, [count, (y1, x1, y2, x2)] ROI boxes in pixels.
    """
    # placeholder
    rois = np.zeros((counts, 4), dtype=np.int32)
    new_rois = []  # tmp
    resize_height, resize_width, _ = image_shape
    new_rois.extend(gt_boxes)
    gt_boxes = np.array(gt_boxes)
    N = gt_boxes.shape[0]
    count = counts - N
    # Generate random ROIs around GT boxes (90% of count)
    rois_per_box = int(count / N)
    for i in range(N):
        gt_y1, gt_x1, gt_y2, gt_x2 = gt_boxes[i]
        cx = (gt_x1 + gt_x2) // 2
        cy = (gt_y1 + gt_y2) // 2
        w = gt_x2 - gt_x1
        h = gt_y2 - gt_y1
        w_scale = [0.8, 1, 1.2]
        h_scale = [0.8, 1, 1.2]
        for hs in h_scale:
            for ws in w_scale:
                new_w = int(w * ws)
                new_h = int(h * hs)
                new_x1 = max(cx - new_w // 2, 0)
                new_x2 = min(cx + new_w // 2, resize_width)
                new_y1 = max(cy - new_h // 2, 0)
                new_y2 = min(cy + new_h // 2, resize_height)
                new_rois.append([new_y1, new_x1, new_y2, new_x2])
        if rois_per_box > 9:
            grid_num = math.ceil(math.sqrt((rois_per_box - 9) / (len(w_scale) * len(h_scale))))
            if grid_num & 1 == 1:
                grid_num += 1
            grid_per_w = w // grid_num
            grid_per_h = h // grid_num
            for w_i in range(grid_num):
                for h_j in range(grid_num):
                    new_cx = gt_x1 + w_i * grid_per_w + grid_per_w // 2
                    new_cy = gt_y1 + h_j * grid_per_h + grid_per_h // 2
                    for hs in h_scale:
                        for ws in w_scale:
                            new_w = int(w * ws)
                            new_h = int(h * hs)
                            new_x1 = max(new_cx - new_w // 2, 0)
                            new_x2 = min(new_cx + new_w // 2, resize_width)
                            new_y1 = max(new_cy - new_h // 2, 0)
                            new_y2 = min(new_cy + new_h // 2, resize_height)
                            new_rois.append([new_y1, new_x1, new_y2, new_x2])
    # Generate random ROIs anywhere in the image
    remaining_count = counts - len(new_rois)
    # To avoid generating boxes with zero area, we generate double what
    # we need and filter out the extra. If we get fewer valid boxes
    # than we need, we loop and try again.
    if remaining_count > 0:
        while True:
            y1y2 = np.random.randint(0, image_shape[0], (remaining_count * 2, 2))
            x1x2 = np.random.randint(0, image_shape[1], (remaining_count * 2, 2))
            # Filter out zero area boxes
            threshold = 1
            y1y2 = y1y2[np.abs(y1y2[:, 0] - y1y2[:, 1]) >=
                        threshold][:remaining_count]
            x1x2 = x1x2[np.abs(x1x2[:, 0] - x1x2[:, 1]) >=
                        threshold][:remaining_count]
            if y1y2.shape[0] == remaining_count and x1x2.shape[0] == remaining_count:
                break

        # Sort on axis 1 to ensure x1 <= x2 and y1 <= y2 and then reshape
        # into x1, y1, x2, y2 order
        x1, x2 = np.split(np.sort(x1x2, axis=1), 2, axis=1)
        y1, y2 = np.split(np.sort(y1y2, axis=1), 2, axis=1)
        global_rois = np.hstack([y1, x1, y2, x2])
        rois[:-remaining_count] = np.array(new_rois)
        rois[-remaining_count:] = global_rois
    else:
        rois[:] = np.array(new_rois)[:counts]
    return rois


def data_generator(dataset, config, shuffle=True, augment=True, random_rois_num=0,
                   batch_size=1, detection_targets=False):
    """A generator that returns images and corresponding target class ids,
    bounding box deltas, and masks.videos45/000000000026

    dataset: The Dataset object to pick data from
    config: The model config object
    shuffle: If True, shuffles the samples before every epoch
    augment: If True, applies image augmentation to images (currently only
             horizontal flips are supported)
    random_rois_num: If > 0 then generate proposals to be used to train the
                 network classifier and mask heads. Useful if training
                 the Mask RCNN part without the RPN.
    batch_size: How many images to return in each call
    detection_targets: If True, generate detection targets (class IDs, bbox
        deltas, and masks). Typically for debugging or visualizations because
        in trainig detection targets are generated by DetectionTargetLayer.

    Returns a Python generator. Upon calling next() on it, the
    generator returns two lists, inputs and outputs. The containtes
    of the lists differs depending on the received arguments:
    inputs list:
    - images: [batch, H, W, C]
    - image_meta: [batch, size of image meta]
    - rpn_match: [batch, N] Integer (1=positive anchor, -1=negative, 0=neutral)
    - rpn_bbox: [batch, N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    - gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs
    - gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
    - gt_masks: [batch, height, width, MAX_GT_INSTANCES]. The height and width
                are those of the image unless use_mini_mask is True, in which
                case they are defined in MINI_MASK_SHAPE.
    - gt_parts: [batch, height, width] of uint8 type. The height and width are
                those of the image

    outputs list: Usually empty in regular training. But if detection_targets
        is True then the outputs list contains target class_ids, bbox deltas,
        and masks.
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
            image_id = image_ids[image_index]  # index for example 0, 1, 2 ..., image_ids is 0, 1, 2 ... shuffled
            # 1. image(input_image): current frame input, shape (512, 512, 3)
            # 2. image_meta(meta_data): shape [10=1+3+4+2], meta include image_id(1), image_shape(3), window(4),
            #    active_class_ids(2), detail in func compose_image_meta
            # mode == "training":
            # 3. gt_class_ids(input_cur_class_ids): GT Class IDs (zero padded), mode is training
            # 4. gt_boxes(input_cur_boxes):GT Boxes in pixels (zero padded)[batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
            # 5. gt_masks(input_cur_mask): GT Masks (zero padded), [batch, height(default 56), width, MAX_GT_INSTANCES]
            # 6. gt_parts(input_cur_part):GT Part [resize_height(IMAGE_MAX_DIM=512), resize_width(IMAGE_MAX_DIM=512)]
            #    the value is 0-19, 0 is bg, 1-19 is the person part label
            image, image_meta, gt_class_ids, gt_boxes, gt_masks, gt_parts, scale = \
                load_image_gt(dataset, config, image_id, augment=augment,
                              use_mini_mask=config.USE_MINI_MASK)
            pre_image_names = dataset.load_pre_image_names(image_id)
            # 7. pre_images(input_pre_images): pre frame image input
            #    pre_masks(input_pre_masks): pre frame mask input
            #    pre_parts(input_pre_parts): pre frame part input
            pre_images, pre_masks, pre_parts, = dataset.load_pre_image_datas(image_id, pre_image_names, config, )
            pre_boxes = dataset.load_pre_image_boxes(image_id, pre_image_names, scale)
            # Skip images that have no instances. This can happen in cases
            # where we train on a subset of classes and the image doesn't
            # have any of the classes we care about.
            if not np.any(gt_class_ids > 0):
                continue
            if len(pre_boxes) == 0:
                continue
            # RPN Targets
            rpn_match, rpn_bbox = build_rpn_targets(image.shape, anchors,
                                                    gt_class_ids, gt_boxes, config)

            # Mask R-CNN Targets
            # random_rois_num = 256
            # detection_targets = True
            if random_rois_num:
                # rpn_rois = generate_random_rois(image.shape, random_rois, pre_boxes)
                rpn_rois = generate_rois(image.shape, random_rois_num, pre_boxes)
                if detection_targets:
                    rois, mrcnn_class_ids, mrcnn_bbox, mrcnn_mask, mrcnn_part = \
                        build_detection_targets(rpn_rois, gt_class_ids, gt_boxes, gt_masks, gt_parts, config)
            # Init batch arrays
            if b == 0:
                batch_rpn_match = np.zeros(
                    [batch_size, anchors.shape[0], 1], dtype=rpn_match.dtype)
                batch_rpn_bbox = np.zeros(
                    [batch_size, config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4], dtype=rpn_bbox.dtype)
                batch_images = np.zeros(
                    (batch_size,) + image.shape, dtype=np.float32)
                batch_image_meta = np.zeros(
                    (batch_size,) + image_meta.shape, dtype=image_meta.dtype)
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
                batch_pre_images = []
                batch_pre_masks = []
                batch_pre_parts = []
                for pre_image in pre_images:
                    batch_pre_images.append(np.zeros((batch_size,) + pre_image.shape, dtype=np.float32))
                for pre_mask in pre_masks:
                    batch_pre_masks.append(np.zeros((batch_size,) + pre_mask.shape, dtype=np.float32))
                for pre_part in pre_parts:
                    batch_pre_parts.append(np.zeros((batch_size,) + pre_part.shape, dtype=np.float32))
                if random_rois_num:
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
            batch_rpn_match[b] = rpn_match[:, np.newaxis]
            batch_rpn_bbox[b] = rpn_bbox
            batch_images[b] = mold_image(image.astype(np.float32), config)
            batch_image_meta[b] = image_meta
            batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids
            batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
            batch_gt_masks[b, :, :, :gt_masks.shape[-1]] = gt_masks
            batch_gt_parts[b, :, :] = gt_parts
            for i, pre_image in enumerate(pre_images):
                batch_pre_images[i][b] = pre_image
            for i, pre_mask in enumerate(pre_masks):
                batch_pre_masks[i][b] = pre_mask
            for i, pre_part in enumerate(pre_parts):
                batch_pre_parts[i][b] = pre_part
            if random_rois_num:
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
                inputs = [batch_images, batch_image_meta,
                          batch_rpn_match, batch_rpn_bbox,
                          batch_gt_class_ids, batch_gt_boxes, batch_gt_masks,
                          batch_gt_parts] + batch_pre_images + batch_pre_masks + batch_pre_parts
                outputs = []

                if random_rois_num:
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
            logging.exception("Error processing image {} in {}, pre_image_names={}".format(
                dataset.image_info[image_id], dataset.get_subset(), pre_image_names))
            error_count += 1
            # if error_count > 5:
            #     raise


class MFP(object):
    """Encapsulates the Mask RCNN model functionality.

    The actual Keras model is in the keras_model property.
    """

    def __init__(self, mode, config, model_dir):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.keras_model = self.build(mode=mode, config=config)

    def build(self, mode, config):
        """Build Mask R-CNN architecture.
            input_shape: The shape of the input image.
            mode: Either "training" or "inference". The inputs and
                outputs of the model differ accordingly.
        """
        assert mode in ['training', 'inference']

        # Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2 ** 4 != int(h / 2 ** 4) or w / 2 ** 4 != int(w / 2 ** 4):
            raise Exception("Image size must be dividable by 2 at least 4 times "
                            "to avoid fractions when downscaling and upscaling.")

        # 1, current frame input, shape (512, 512, 3)
        input_image = KL.Input(
            shape=config.IMAGE_SHAPE.tolist(), name="input_image")
        # 2, meta data, shape [10=1+3+4+2], meta include image_id(1), image_shape(3), window(4), active_class_ids(2),
        # detail in func compose_image_meta
        input_image_meta = KL.Input(shape=[None], name="input_image_meta")
        if mode == "training":
            # Detection GT (class IDs, bounding boxes, and masks)
            # 3. GT Class IDs (zero padded)
            input_cur_class_ids = KL.Input(
                shape=[None], name="input_cur_class_ids", dtype=tf.int32)
            # 4. GT Boxes in pixels (zero padded)
            # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates
            input_cur_boxes = KL.Input(
                shape=[None, 4], name="input_cur_boxes", dtype=tf.float32)
            # Normalize coordinates
            h, w = K.shape(input_image)[1], K.shape(input_image)[2]
            image_scale = K.cast(K.stack([h, w, h, w], axis=0), tf.float32)
            gt_boxes = KL.Lambda(lambda x: x / image_scale)(input_cur_boxes)  # the coordinate is normalized to 1
            # 5. GT Masks (zero padded)
            # [batch, height, width, MAX_GT_INSTANCES]
            if config.USE_MINI_MASK:
                input_cur_mask = KL.Input(
                    shape=[config.MINI_MASK_SHAPE[0],  # 56
                           config.MINI_MASK_SHAPE[1], None],
                    name="input_cur_mask", dtype=bool)
            else:
                input_cur_mask = KL.Input(
                    shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None],
                    name="input_cur_mask", dtype=bool)
            # 6. GT Part
            input_cur_part = KL.Input(
                shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1]],
                name="input_cur_part", dtype=tf.uint8)
            # 8, RPN GT
            input_rpn_match = KL.Input(
                shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
            input_rpn_bbox = KL.Input(
                shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)
        # 7, pre frame input(pre image+mask+part)
        input_pre_images = []
        input_pre_masks = []
        input_pre_parts = []
        if config.IS_PRE_IMAGE:
            for i in range(config.PRE_MULTI_FRAMES):
                # pre image
                input_pre_images.append(KL.Input(shape=config.PRE_IMAGE_SHAPE, name="input_pre_image_" + str(i)))
        if config.IS_PRE_MASK:
            for i in range(config.PRE_MULTI_FRAMES):
                # pre mask
                input_pre_masks.append(
                    KL.Input(shape=[config.PRE_IMAGE_SHAPE[0], config.PRE_IMAGE_SHAPE[1], 1],
                             name="input_pre_mask_" + str(i)))
        if config.IS_PRE_PART:
            for i in range(config.PRE_MULTI_FRAMES):
                # pre part
                input_pre_parts.append(
                    KL.Input(shape=[config.PRE_IMAGE_SHAPE[0], config.PRE_IMAGE_SHAPE[1], 20],
                             name="input_pre_part_" + str(i)))
        # # 8, rois from pre frames
        # # Ignore predicted ROIs and use ROIs provided as an input.
        # input_rois = KL.Input(shape=[config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4],
        #                       name="input_roi", dtype=tf.float32)
        # # Normalize coordinates
        # target_rois = KL.Lambda(lambda x: norm_boxes_graph(
        #     x, K.shape(input_image)[1:3]))(input_rois)

        # above is the model input

        # Build the shared convolutional layers.
        # Bottom-up Layers
        # Returns a list of the last layers of each stage, 5 in total.
        # Don't create the thead (stage 5), so we pick the 4th item in the list.
        if config.RECURRENT_UNIT == 'lstm':
            feature_pre_images = conv_lstm_unit(input_pre_images, name="feature_pre_images", initial_state=None)
            feature_pre_masks = conv_lstm_unit(input_pre_masks, name="feature_pre_masks", initial_state=None)
            feature_pre_parts = conv_lstm_unit(input_pre_parts, name="feature_pre_parts", initial_state=None)
        elif config.RECURRENT_UNIT == 'gru':
            # deal with pre images
            feature_pre_images = []
            if config.IS_PRE_IMAGE:
                feature_pre_images.append(
                    conv3d_gru2d_unit(input_pre_images, filter=config.RECURRENT_FILTER, name="feature_pre_images"))
            # deal with pre masks
            feature_pre_masks = []
            if config.IS_PRE_MASK:
                feature_pre_masks.append(
                    conv3d_gru2d_unit(input_pre_masks, filter=config.RECURRENT_FILTER, name="feature_pre_masks"))
            # deal with pre parts
            feature_pre_parts = []
            if config.IS_PRE_PART:
                feature_pre_parts.append(
                    conv3d_gru2d_unit(input_pre_parts, filter=config.RECURRENT_FILTER, name="feature_pre_parts"))
            features = feature_pre_images + feature_pre_masks + feature_pre_parts
            if len(features) > 1:
                feature_merge = conv3d_gru2d_unit(features,
                                                  filter=config.RECURRENT_FILTER, name="merge")
            else:
                feature_merge = features[0]
        C1, C2, C3, C4, C5 = deeplab_resnet(input_image, 'resnet50')
        coarse_feature = global_parsing_encoder(C5)
        fine_feature = global_parsing_decoder(coarse_feature, feature_merge)
        # global parsing branch
        global_parsing_map = global_parsing_graph(fine_feature, config.NUM_PART_CLASS)

        rpn_feature_map = KL.Conv2D(256, (3, 3), activation='relu', padding='same',
                                    name='mrcnn_share_rpn_conv1')(fine_feature)
        rpn_feature_map = KL.Conv2D(256, (3, 3), activation='relu', padding='same',
                                    name='mrcnn_share_rpn_conv2')(rpn_feature_map)  # shape [batch, 128, 128, 256]

        mrcnn_feature_map = KL.Conv2D(256, (3, 3), activation='relu', padding='same',
                                      name='mrcnn_share_recog_conv1')(fine_feature)
        mrcnn_feature_map = KL.Conv2D(256, (3, 3), activation='relu', padding='same',
                                      name='mrcnn_share_recog_conv2')(mrcnn_feature_map)  # shape [batch, 128, 128, 256]

        # Generate Anchors
        self.anchors = util.generate_anchors(config.RPN_ANCHOR_SCALES,
                                             config.RPN_ANCHOR_RATIOS,
                                             config.BACKBONE_SHAPES[0],
                                             config.BACKBONE_STRIDES[0],
                                             config.RPN_ANCHOR_STRIDE)

        # RPN Model
        rpn_class_logits, rpn_class, rpn_bbox = rpn_graph(rpn_feature_map,
                                                          len(config.RPN_ANCHOR_RATIOS) * len(config.RPN_ANCHOR_SCALES),
                                                          config.RPN_ANCHOR_STRIDE)

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        proposal_count = config.POST_NMS_ROIS_TRAINING if mode == "training" \
            else config.POST_NMS_ROIS_INFERENCE
        pre_proposal_count = config.PRE_NMS_ROIS_TRAINING if mode == "training" \
            else config.PRE_NMS_ROIS_INFERENCE
        rpn_rois = ProposalLayer(proposal_count=proposal_count,
                                 pre_proposal_count=pre_proposal_count,
                                 nms_threshold=config.RPN_NMS_THRESHOLD,
                                 name="ROI",
                                 anchors=self.anchors,
                                 config=config)([rpn_class, rpn_bbox])
        # rpn_rois: [batch, proposal_count, 4], the coordinate is normalized to 1
        if mode == "training":
            # Class ID mask to mark class IDs supported by the dataset the image
            # came from.
            _, _, _, active_class_ids = KL.Lambda(lambda x: parse_image_meta_graph(x),
                                                  mask=[None, None, None, None])(input_image_meta)
            if not config.USE_RPN_ROIS:
                # Ignore predicted ROIs and use ROIs provided as an input.
                input_rois = KL.Input(shape=[config.POST_NMS_ROIS_TRAINING, 4],
                                      name="input_roi", dtype=np.int32)
                # Normalize coordinates
                target_rois = KL.Lambda(lambda x: norm_boxes_graph(
                    x, K.shape(input_image)[1:3]))(input_rois)
            else:
                target_rois = rpn_rois  # shape [batch, proposal_count, 4], the coordinate is normalized to 1
            # Generate detection targets
            # Subsamples proposals and generates target outputs for training
            # Note that proposal class IDs, gt_boxes, and gt_masks are zero
            # padded. Equally, returned rois and targets are zero padded.
            # rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized to 1, TRAIN_ROIS_PER_IMAGE=128
            # target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
            # target_bbox(target_deltas): [batch, TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw))]
            # target_mask: [batch, TRAIN_ROIS_PER_IMAGE, m_height=28, m_width=28)
            rois, target_class_ids, target_bbox, target_mask = \
                DetectionTargetLayer(config, name="proposal_targets")([
                    target_rois, input_cur_class_ids, gt_boxes, input_cur_mask])

            # Network Heads
            # TODO: verify that this handles zero padded ROIs
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = \
                fpn_classifier_graph(rois, mrcnn_feature_map, config.IMAGE_SHAPE,
                                     config.POOL_SIZE, config.NUM_CLASSES)

            mrcnn_mask = build_fpn_mask_graph(rois, mrcnn_feature_map,
                                              config.IMAGE_SHAPE,
                                              config.MASK_POOL_SIZE,
                                              config.NUM_CLASSES)

            # TODO: clean up (use tf.identify if necessary)
            output_rois = KL.Lambda(lambda x: x * 1, name="output_rois")(rois)

            # Losses
            rpn_class_loss = KL.Lambda(lambda x: rpn_class_loss_graph(*x), name="rpn_class_loss")(
                [input_rpn_match, rpn_class_logits])
            rpn_bbox_loss = KL.Lambda(lambda x: rpn_bbox_loss_graph(config, *x), name="rpn_bbox_loss")(
                [input_rpn_bbox, input_rpn_match, rpn_bbox])
            class_loss = KL.Lambda(lambda x: mrcnn_class_loss_graph(*x), name="mrcnn_class_loss")(
                [target_class_ids, mrcnn_class_logits, active_class_ids])
            bbox_loss = KL.Lambda(lambda x: mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss")(
                [target_bbox, target_class_ids, mrcnn_bbox])
            mask_loss = KL.Lambda(lambda x: mrcnn_mask_loss_graph(*x), name="mrcnn_mask_loss")(
                [target_mask, target_class_ids, mrcnn_mask])
            global_parsing_loss = KL.Lambda(lambda x: mrcnn_global_parsing_loss_graph(config.NUM_PART_CLASS, *x),
                                            name="mrcnn_global_parsing_loss")(
                [input_cur_part, global_parsing_map])

            # Model
            inputs = [input_image, input_image_meta,
                      input_rpn_match, input_rpn_bbox,
                      input_cur_class_ids, input_cur_boxes,
                      input_cur_mask,
                      input_cur_part] + input_pre_images + input_pre_masks + input_pre_parts
            if not config.USE_RPN_ROIS:
                inputs.append(input_rois)
            outputs = [rpn_class_logits, rpn_class, rpn_bbox,
                       mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask,
                       rpn_rois, output_rois,
                       rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss,
                       global_parsing_loss]
            model = KM.Model(inputs, outputs, name='parsing_rcnn')
        else:
            # Network Heads
            # Proposal classifier and BBox regressor heads
            target_rois = rpn_rois
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = \
                fpn_classifier_graph(target_rois, mrcnn_feature_map, config.IMAGE_SHAPE,
                                     config.POOL_SIZE, config.NUM_CLASSES)

            # Detections
            # target_rois = KL.Lambda(lambda x: x * 1, name="output_rois")(target_rois)
            # output is [batch, num_detections(default 100), (y1, x1, y2, x2, class_id, score)] in image coordinates
            detections = DetectionLayer(config, name="mrcnn_detection")(
                [target_rois, mrcnn_class, mrcnn_bbox, input_image_meta])

            # Convert boxes to normalized coordinates
            # TODO: let DetectionLayer return normalized coordinates to avoid
            #       unnecessary conversions
            h, w = config.IMAGE_SHAPE[:2]
            detection_boxes = KL.Lambda(
                lambda x: x[..., :4] / np.array([h, w, h, w]))(detections)

            # Create masks for detections
            mrcnn_mask = build_fpn_mask_graph(detection_boxes, mrcnn_feature_map,
                                              config.IMAGE_SHAPE,
                                              config.MASK_POOL_SIZE,
                                              config.NUM_CLASSES)

            # global parsing branch
            global_parsing_prob = KL.Lambda(lambda x: post_processing_graph(*x))([global_parsing_map, input_image])

            model = KM.Model(
                [input_image, input_image_meta] + input_pre_images + input_pre_masks + input_pre_parts,
                [detections, mrcnn_class, mrcnn_bbox, mrcnn_mask,
                 rpn_rois, rpn_class, rpn_bbox,
                 global_parsing_prob], name='parsing_rcnn')

        # Add multi-GPU support.
        if config.GPU_COUNT > 1:
            from utils.parallel_model import ParallelModel
            model = ParallelModel(model, config.GPU_COUNT)
        import platform
        sys = platform.system()
        if sys == "Windows":
            if self.mode == "training":
                plot_model(model, "mfp_training.jpg")
            else:
                plot_model(model, "mfp_inference.png")
        return model

    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            log_dir: The directory where events and weights are saved
            checkpoint_path: the path to the last checkpoint file
        """
        # Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            return None, None
        # Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("parsing_rcnn"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            return dir_name, None
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return dir_name, checkpoint

    def load_weights(self, filepath, by_name=False, exclude_pattern=None):
        """Modified version of the correspoding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exlude: list of layer names to excluce
        """
        import h5py
        import keras
        if keras.__version__ > "2.2.0":
            from keras.engine import saving as s
        else:
            from keras.engine import topology as s

        if exclude_pattern:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        keras_model = self.keras_model
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model") \
            else keras_model.layers

        # exclude some layers
        if exclude_pattern:
            layers = filter(lambda l: exclude_pattern.match(l.name) == None, layers)

        layers_name = [l.name for l in layers]
        print("load model", filepath)

        if by_name:
            # s.load_weights_from_hdf5_group_by_name(f, layers, skip_mismatch=True)
            s.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            s.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()

        # Update the log directory
        self.set_log_dir(filepath)

    def compile(self, learning_rate, momentum):
        """Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        """
        # Optimizer object
        optimizer = keras.optimizers.SGD(lr=learning_rate, momentum=momentum,
                                         clipnorm=5.0)
        # Add Losses
        # First, clear previously set losses to avoid duplication
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}
        loss_names = ["rpn_class_loss", "rpn_bbox_loss",
                      "mrcnn_class_loss", "mrcnn_bbox_loss", "mrcnn_mask_loss",
                      "mrcnn_global_parsing_loss"]
        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            if layer.output in self.keras_model.losses:
                continue
            self.keras_model.add_loss(
                tf.reduce_mean(layer.output))

        # Add L2 Regularization
        reg_losses = [keras.regularizers.l2(self.config.WEIGHT_DECAY)(w)
                      for w in self.keras_model.trainable_weights
                      if ('gamma' not in w.name) and ('beta' not in w.name) and ('bias' not in w.name)]

        self.keras_model.add_loss(tf.add_n(reg_losses))

        # Compile
        self.keras_model.compile(optimizer=optimizer, loss=[
                                                               None] * len(self.keras_model.outputs))

        # Add metrics for losses
        for name in loss_names:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            self.keras_model.metrics_tensors.append(tf.reduce_mean(
                layer.output, keep_dims=True))

    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """
        # Print message on the first call (but not on recursive calls)
        if verbose > 0 and keras_model is None:
            log("Selecting layers to train")

        keras_model = keras_model or self.keras_model

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model") \
            else keras_model.layers

        for layer in layers:
            # Is the layer a model?
            if layer.__class__.__name__ == 'Model':
                print("In model: ", layer.name)
                self.set_trainable(
                    layer_regex, keras_model=layer, indent=indent + 4)
                continue

            if not layer.weights:
                continue
            # Is it trainable?
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            # Print trainble layer names
            if trainable and verbose > 0:
                log("{}{:20}   ({})".format(" " * indent, layer.name,
                                            layer.__class__.__name__))

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
            regex = r".*/\w+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/parsing\_rcnn\_\w+(\d{4})\.h5"
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
                                            "parsing_rcnn_" + self.config.NAME.lower() +
                                            "_epoch{epoch:03d}_loss{loss:.3f}_valloss{val_loss:.3f}.h5")
        if not os.path.exists(os.path.dirname(self.checkpoint_path)):
            os.makedirs(os.path.dirname(self.checkpoint_path))

        self.log_path = os.path.join(self.log_dir, "logs", "parsing_rcnn_{}_{:%Y%m%dT%H%M}.csv".format(
            self.config.NAME.lower(), now))
        if not os.path.exists(os.path.dirname(self.log_path)):
            os.makedirs(os.path.dirname(self.log_path))

        self.tensorboard_dir = os.path.join(self.log_dir, "tensorboard")

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
            "rpn": r"(rpn\_.*)",
            "roi_process_head": r"(mrcnn\_bbox\_.*)|(mrcnn\_class\_.*)|(mrcnn\_mask\_.*)|(mrcnn\_share\_.*)",
            "mask_heads": r"(mrcnn\_bbox\_.*)|(rpn\_.*)|(mrcnn\_class\_.*)|(mrcnn\_mask\_.*)|(mrcnn\_share\_.*)",
            "heads": r"(mrcnn\_.*)|(rpn\_.*)",
            # From a specific Resnet stage and up
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)",
            # All layers
            "all": ".*",
            "first_stage": r"(mrcnn\_bbox\_.*)|(rpn\_.*)|(mrcnn\_class\_.*)|(mrcnn\_mask\_.*)|(mrcnn\_share\_.*)",
            "second_stage": r"(mrcnn\_.*)|(rpn\_.*)",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Data generators
        print("get train generator")
        train_generator = data_generator(train_dataset, self.config, shuffle=True,
                                         random_rois_num=0, batch_size=self.config.BATCH_SIZE)
        print("get val generator")
        val_generator = data_generator(val_dataset, self.config, shuffle=True,
                                       random_rois_num=0, batch_size=self.config.BATCH_SIZE)

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
        self.set_trainable(layers, verbose=0)
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
        )
        self.epoch = max(self.epoch, epochs)

    def ancestor(self, tensor, name, checked=None):
        """Finds the ancestor of a TF tensor in the computation graph.
        tensor: TensorFlow symbolic tensor.
        name: Name of ancestor tensor to find
        checked: For internal use. A list of tensors that were already
                 searched to avoid loops in traversing the graph.
        """
        checked = checked if checked is not None else []
        # Put a limit on how deep we go to avoid very long loops
        if len(checked) > 500:
            return None
        # Convert name to a regex and allow matching a number prefix
        # because Keras adds them automatically
        if isinstance(name, str):
            name = re.compile(name.replace("/", r"(\_\d+)*/"))

        parents = tensor.op.inputs
        for p in parents:
            if p in checked:
                continue
            if bool(re.fullmatch(name, p.name)):
                return p
            checked.append(p)
            a = self.ancestor(p, name, checked)
            if a is not None:
                return a
        return None

    def find_trainable_layer(self, layer):
        """If a layer is encapsulated by another layer, this function
        digs through the encapsulation and returns the layer that holds
        the weights.
        """
        if layer.__class__.__name__ == 'TimeDistributed':
            return self.find_trainable_layer(layer.layer)
        return layer

    def get_trainable_layers(self):
        """Returns a list of layers that have weights."""
        layers = []
        # Loop through all layers
        for l in self.keras_model.layers:
            # If layer is a wrapper, find inner trainable layer
            l = self.find_trainable_layer(l)
            # Include layer if it has weights
            if l.get_weights():
                layers.append(l)
        return layers

    def mold_inputs(self, images, isopencv=False):
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.
        Args:
            images: List of image matricies [height,width,depth]. Images can have
                different sizes.

        Returns: 3 Numpy matricies
            molded_images: [batch, h, w, 3]. Images resized and normalized.
            image_metas: [batch, length of meta data(default ]. Details about each image.
            windows: [batch, (y1, x1, y2, x2)]. The portion of the image that has the
                original image (padding excluded).
        """
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            # Resize image to fit the model expected size
            # TODO: move resizing to mold_image()
            molded_image, window, scale, padding = util.resize_image(
                image,
                min_dim=self.config.IMAGE_MIN_DIM,
                max_dim=self.config.IMAGE_MAX_DIM,
                padding=self.config.IMAGE_PADDING, isopencv=isopencv)
            molded_image = mold_image(molded_image, self.config)
            # Build image_meta
            image_meta = compose_image_meta(
                0, image.shape, window,
                np.zeros([self.config.NUM_CLASSES], dtype=np.int32))
            # Append
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        # Pack into arrays
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows

    def unmold_detections(self, detections, mrcnn_mask, mrcnn_global_parsing, image_shape, window, isopencv=False):
        """Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the
        application.

        detections: [N, (y1, x1, y2, x2, class_id, score)]
        mrcnn_mask: [N, height, width, num_classes]
        mrcnn_global_parsing: [resized_height, resized_width, num_classes]
        image_shape: [height, width, depth] Original size of the image before resizing
        window: [y1, x1, y2, x2] Box in the image where the real image is
                excluding the padding.

        Returns:
        boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        masks: [height, width, num_instances] Instance masks
        """
        # How many detections do we have?
        # Detections array is padded with zeros. Find the first class_id == 0.
        zero_ix = np.where(detections[:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        scores = detections[:N, 5]
        masks = mrcnn_mask[np.arange(N), :, :, class_ids]

        # Compute scale and shift to translate coordinates to image domain.
        h_scale = image_shape[0] / (window[2] - window[0])
        w_scale = image_shape[1] / (window[3] - window[1])
        scale = min(h_scale, w_scale)
        shift = window[:2]  # y, x
        scales = np.array([scale, scale, scale, scale])
        shifts = np.array([shift[0], shift[1], shift[0], shift[1]])

        # Translate bounding boxes to image domain
        boxes = np.multiply(boxes - shifts, scales).astype(np.int32)

        # Filter out detections with zero area. Often only happens in early
        # stages of training when the network weights are still a bit random.
        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            N = class_ids.shape[0]

        # Resize masks to original image size and set boundary threshold.
        full_masks = []
        for i in range(N):
            # Convert neural network mask to full size mask
            full_mask = util.unmold_mask(masks[i], boxes[i], image_shape, isopencv=False)  # use opencv is slow
            full_masks.append(full_mask)
        full_masks = np.stack(full_masks, axis=-1) \
            if full_masks else np.empty((0,) + masks.shape[1:3])

        global_parsing = mrcnn_global_parsing[window[0]:window[2], window[1]:window[3], :]
        global_parsing = skimage.transform.resize(global_parsing, (image_shape[0], image_shape[1]), mode="constant")

        return boxes, class_ids, scores, full_masks, global_parsing

    def detect(self, images, pre_images, pre_masks, pre_parts, verbose=0, isopencv=False):
        """Runs the detection pipeline.
        Args:
            images: List of images, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
            rois: [N, (y1, x1, y2, x2)] detection bounding boxes
            class_ids: [N] int class IDs
            scores: [N] float probability scores for the class IDs
            masks: [H, W, N] instance binary masks
        """
        assert self.mode == "inference", "Create model in inference mode."
        assert len(
            images) == self.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

        if verbose:
            log("Processing {} images".format(len(images)))
            for image in images:
                log("image", image)
        # Mold inputs to format expected by the neural network
        molded_images, image_metas, windows = self.mold_inputs(images, isopencv=isopencv)
        # random_rois = 256
        # rpn_rois = generate_random_rois(images[0].shape, random_rois, pre_boxes)
        if verbose:
            log("molded_images", molded_images)
            log("image_metas", image_metas)
        # Run object detection
        detections, mrcnn_class, mrcnn_bbox, mrcnn_mask, \
        rpn_rois, rpn_class, rpn_bbox, mrcnn_global_parsing_prob = \
            self.keras_model.predict(
                [molded_images, image_metas] + pre_images + pre_masks + pre_parts,
                verbose=0)
        # Process detections
        results = []
        for i, image in enumerate(images):
            final_boxes, final_class_ids, final_scores, final_masks, final_globals = \
                self.unmold_detections(detections[i], mrcnn_mask[i], mrcnn_global_parsing_prob[i],
                                       image.shape, windows[i], isopencv=isopencv)
            results.append({
                "boxes": final_boxes,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
                "global_parsing": final_globals
            })
        return results


############################################################
#  Data Formatting
############################################################

def compose_image_meta(image_id, image_shape, window, active_class_ids):
    """Takes attributes of an image and puts them in one 1D array. Use
    parse_image_meta() to parse the values back.

    image_id: An int ID of the image. Useful for debugging.
    image_shape: [height, width, channels]
    window: (y1, x1, y2, x2) in pixels. The area of the image where the real
            image is (excluding the padding)
    active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on images from multiple datasets
        where not all classes are present in all datasets.
    """
    meta = np.array(
        [image_id] +  # size=1
        list(image_shape) +  # size=3
        list(window) +  # size=4 (y1, x1, y2, x2) in image cooredinates
        list(active_class_ids)  # size=num_classes
    )
    return meta


# Two functions (for Numpy and TF) to parse image_meta tensors.
def parse_image_meta(meta):
    """Parses an image info Numpy array to its components.
    See compose_image_meta() for more details.
    """
    image_id = meta[:, 0]
    image_shape = meta[:, 1:4]
    window = meta[:, 4:8]  # (y1, x1, y2, x2) window of image in in pixels
    active_class_ids = meta[:, 8:]
    return image_id, image_shape, window, active_class_ids


def parse_image_meta_graph(meta):
    """Parses a tensor that contains image attributes to its components.
    See compose_image_meta() for more details.

    meta: [batch, meta length] where meta length depends on NUM_CLASSES
    """
    image_id = meta[:, 0]
    image_shape = meta[:, 1:4]
    window = meta[:, 4:8]
    active_class_ids = meta[:, 8:]
    return [image_id, image_shape, window, active_class_ids]


def mold_image(images, config):
    """Takes RGB images with 0-255 values and subtraces
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return images.astype(np.float32) - config.MEAN_PIXEL


def unmold_image(normalized_images, config):
    """Takes a image normalized with mold() and returns the original."""
    return (normalized_images + config.MEAN_PIXEL).astype(np.uint8)


############################################################
#  Miscellenous Graph Functions
############################################################

def trim_zeros_graph(boxes, name=None):
    """Often boxes are represented with matricies of shape [N, 4] and
    are padded with zeros. This removes zero boxes.

    boxes: [N, 4] matrix of boxes.
    non_zeros: [N] a 1D boolean mask identifying the rows to keep
    """
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros


def batch_pack_graph(x, counts, num_rows):
    """Picks different number of values from each row
    in x depending on the values in counts.
    """
    outputs = []
    for i in range(num_rows):
        outputs.append(x[i, :counts[i]])
    return tf.concat(outputs, axis=0)


def norm_boxes_graph(boxes, shape):
    """Converts boxes from pixel coordinates to normalized coordinates.
    boxes: [..., (y1, x1, y2, x2)] in pixel coordinates
    shape: [..., (height, width)] in pixels
    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.
    Returns:
        [..., (y1, x1, y2, x2)] in normalized coordinates
    """
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.divide(boxes - shift, scale)


def denorm_boxes_graph(boxes, shape):
    """Converts boxes from normalized coordinates to pixel coordinates.
    boxes: [..., (y1, x1, y2, x2)] in normalized coordinates
    shape: [..., (height, width)] in pixels
    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.
    Returns:
        [..., (y1, x1, y2, x2)] in pixel coordinates
    """
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.cast(tf.round(tf.multiply(boxes, scale) + shift), tf.int32)
