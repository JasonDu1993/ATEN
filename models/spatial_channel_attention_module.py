from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, \
    Conv2D, Add, Activation, Lambda, Input
from keras import backend as K
from keras.activations import sigmoid


def attach_attention_module(net, attention_module):
    if attention_module == 'se_block':  # SE_block
        net = se_block(net)
    elif attention_module == 'cbam_block':  # CBAM_block
        net = cbam_block(net)
    else:
        raise Exception("'{}' is not supported attention module!".format(attention_module))

    return net


def se_block(input_feature, ratio=8):
    """Contains the implementation of Squeeze-and-Excitation(SE) block.
    As described in https://arxiv.org/abs/1709.01507.
    """

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature._keras_shape[channel_axis]

    se_feature = GlobalAveragePooling2D()(input_feature)
    se_feature = Reshape((1, 1, channel))(se_feature)
    assert se_feature._keras_shape[1:] == (1, 1, channel)
    se_feature = Dense(channel // ratio,
                       activation='relu',
                       kernel_initializer='he_normal',
                       use_bias=True,
                       bias_initializer='zeros')(se_feature)
    assert se_feature._keras_shape[1:] == (1, 1, channel // ratio)
    se_feature = Dense(channel,
                       activation='sigmoid',
                       kernel_initializer='he_normal',
                       use_bias=True,
                       bias_initializer='zeros')(se_feature)
    assert se_feature._keras_shape[1:] == (1, 1, channel)
    if K.image_data_format() == 'channels_first':
        se_feature = Permute((3, 1, 2))(se_feature)

    se_feature = multiply([input_feature, se_feature])
    return se_feature


def cbam_block(cbam_feature, ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """

    cbam_feature_channel = channel_attention(cbam_feature, ratio)
    cbam_feature_spatial = spatial_attention(cbam_feature_channel)
    return cbam_feature_spatial


def channel_attention(input_feature, ratio=8):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature._keras_shape[channel_axis]

    avg_pool = GlobalAveragePooling2D()(input_feature)
    max_pool = GlobalMaxPooling2D()(input_feature)
    cocat_feature = Concatenate()([avg_pool, max_pool])
    x = Reshape((1, 1, 2 * channel))(cocat_feature)
    fc1 = Dense(channel // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True,
                bias_initializer='zeros')(x)
    fc2 = Dense(channel, activation="sigmoid", kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')(
        fc1)

    if K.image_data_format() == "channels_first":
        fc2 = Permute((3, 1, 2))(fc2)
    x = multiply([input_feature, fc2])
    return x


def spatial_attention(input_feature):
    kernel_size = 7

    if K.image_data_format() == "channels_first":
        channel = input_feature._keras_shape[1]
        cbam_feature = Permute((2, 3, 1))(input_feature)
    else:
        channel = input_feature._keras_shape[-1]
        cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool._keras_shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool._keras_shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat._keras_shape[-1] == 2
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)
    assert cbam_feature._keras_shape[-1] == 1

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])


if __name__ == '__main__':
    import numpy as np
    import tensorflow as tf
    import os

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # kvar = K.reshape(K.variable(np.arange(64), dtype="int32"), shape=(1, 2, 2, 16))  # 默认dtype="float32"
    # kvar = Reshape((4, 4, 4))(kvar)
    # print(kvar.shape)
    kvar = Input(shape=(4, 4, 4))
    kvar = Conv2D(8, (3, 3), padding="SAME")(kvar)
    print(kvar.shape)
    # a = K.eval(kvar)
    # print(a)
    x = cbam_block(kvar, ratio=2)
    print(x)
    # print(K.eval(x))
