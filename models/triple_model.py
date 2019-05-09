# -*- coding: utf-8 -*-
# @Time    : 2019/4/19 10:13
# @Author  : Jason
# @Email   : 1358681631@qq.com
# @File    : triple_model.py.py
# @Software: PyCharm

import keras.backend as K
from keras.layers import Input, Conv2D, AtrousConv2D, BatchNormalization, ReLU, Reshape, Activation
from keras.layers import Add, Deconv2D, Dense, Lambda, Concatenate, Multiply
from keras.layers import Conv3D, Deconv3D
from keras.layers import LSTM, ConvLSTM2D, ConvLSTM2DCell
from models.convolutional_recurrent import ConvGRU2D
from keras.models import Model
from keras.utils.vis_utils import plot_model
import tensorflow as tf
from configs.vip import VIPDataset
from configs.vip import VideoModelConfig

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


class TripleCNN(object):
    def __init__(self, mode, input_shape, config):
        assert mode in ['training', 'inference']
        self.mode = mode
        self.input_shape = input_shape
        self.keras_model = self.build(mode, input_shape, config)

    def build(self, mode, input_shape, config=None, image_nums=5):
        if not config:
            config = VideoModelConfig()
        assert mode in ["training", "inference"]
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
        if mode == "training":

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

            global_parsing_loss = KL.Lambda(lambda x: mrcnn_global_parsing_loss_graph(config.NUM_PART_CLASS, *x),
                                            name="mrcnn_global_parsing_loss")(
                [input_gt_part, global_parsing_map])

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

            # Model
            inputs = [input_image, input_image_key1, input_image_key2, input_image_key3,
                      input_key_identity, input_image_meta, input_rpn_match, input_rpn_bbox,
                      input_gt_class_ids, input_gt_boxes, input_gt_masks, input_gt_part]
            if not config.USE_RPN_ROIS:
                inputs.append(input_rois)
            outputs = [rpn_class_logits, rpn_class, rpn_bbox,
                       mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask,
                       rpn_rois, output_rois,
                       rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss,
                       global_parsing_loss]
            model = KM.Model(inputs, outputs, name='aten')
        else:
            # Network Heads
            # Proposal classifier and BBox regressor heads
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = \
                fpn_classifier_graph(rpn_rois, mrcnn_feature_map, config.IMAGE_SHAPE,
                                     config.POOL_SIZE, config.NUM_CLASSES)

            # Detections
            # output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in image coordinates
            detections = DetectionLayer(config, name="mrcnn_detection")(
                [rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta])

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

            global_parsing_prob = KL.Lambda(lambda x: post_processing_graph(*x))([global_parsing_map, input_image])

            model = KM.Model([input_image, input_image_key1, input_image_key2, input_image_key3,
                              input_key_identity, input_image_meta],
                             [detections, mrcnn_class, mrcnn_bbox,
                              mrcnn_mask, rpn_rois, rpn_class, rpn_bbox, global_parsing_prob],
                             name='aten')

        # Add multi-GPU support.
        if config.GPU_COUNT > 1:
            from utils.parallel_model import ParallelModel
            model = ParallelModel(model, config.GPU_COUNT)
        import platform
        sys = platform.system()
        # if sys == "Windows":
        if self.mode == "training":
            plot_model(model, "aten_training.jpg")
        else:
            plot_model(model, "aten_test.png")
        return model

    def train(self, lr, batch_size, model, input_shape, annotation_path, image_path, anchors, num_classes, model_name,
              log_dir='./checkpoints/csvlogs/'):
        """retrain/fine-tune the model"""
        model.compile(optimizer=Adam(lr=lr), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})
        lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, verbose=1, mode='min')
        logging = TensorBoard(log_dir=log_dir)
        checkpointer = ModelCheckpoint(
            filepath='./checkpoints/models/' + model_name + '.{epoch:03d}-{loss:.3f}-{val_loss:.3f}.h5',
            verbose=2,
            monitor='val_loss',
            save_weights_only=True,
            save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=1, mode='auto')
        timestamp = time.time()
        csv_logger = CSVLogger('./checkpoints/csvlogs/' + model_name + str(timestamp) + '.log')

        val_split = 0.1
        with open(annotation_path) as f:
            lines = f.readlines()
        np.random.seed(10101)
        np.random.shuffle(lines)
        np.random.seed(None)
        num_val = int(len(lines) * val_split)
        num_train = len(lines) - num_val
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        image_data, box_data = get_training_data(annotation_path, image_path, "./datas/train.npz", input_shape,
                                                 max_boxes=20)
        image_data = image_data / 255.0
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        model.fit([image_data, *y_true],
                  np.zeros(len(image_data)),
                  validation_split=.1,
                  batch_size=batch_size,
                  epochs=100,
                  callbacks=[logging, checkpointer, csv_logger, early_stopping, lr_reduce])


if __name__ == '__main__':
    TripleCNN(mode="training", input_shape=(512, 512, 256))
