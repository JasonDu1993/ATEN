from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, \
    Conv2D, Add, Activation, Lambda, Input, add, Dot
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


def cbam_block(cbam_feature, attn_type="channel_attention", ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """
    assert attn_type in ["se", "channel_attention"]
    if attn_type == "se":
        cbam_feature_channel = se_block(cbam_feature, ratio)
    elif attn_type == "channel_attention":
        cbam_feature_channel = channel_attention(cbam_feature, ratio)
    else:
        cbam_feature_channel = channel_attention(cbam_feature, ratio)
    cbam_feature_spatial = spatial_attention(cbam_feature_channel)
    return cbam_feature_spatial


def cbam_block_parallel(cbam_feature, attn_type="channel_attention", ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """
    assert attn_type in ["se", "channel_attention"]
    if attn_type == "se":
        cbam_feature_channel = se_block(cbam_feature, ratio)
    elif attn_type == "channel_attention":
        cbam_feature_channel = channel_attention(cbam_feature, ratio)
    else:
        cbam_feature_channel = channel_attention(cbam_feature, ratio)
    cbam_feature_spatial = spatial_attention(cbam_feature)
    final_feature = add([cbam_feature, cbam_feature_channel, cbam_feature_spatial])
    return final_feature

def cbam_block_parallel2(cbam_feature, attn_type="se", ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """
    assert attn_type in ["se", "channel_attention"]
    if attn_type == "se":
        cbam_feature_channel = se_block(cbam_feature, ratio)
    elif attn_type == "channel_attention":
        cbam_feature_channel = channel_attention(cbam_feature, ratio)
    else:
        cbam_feature_channel = channel_attention(cbam_feature, ratio)
    cbam_feature_spatial = spatial_attention(cbam_feature)
    attn_map = multiply([cbam_feature_channel, cbam_feature_spatial])
    final_feature = add([cbam_feature, attn_map])
    return final_feature

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


def position_attention_module(inputs, name, ratio=8):
    """ Position attention module"""
    # Ref from SAGAN
    m_batchsize, height, width, in_dim = K.int_shape(inputs)
    f = in_dim // ratio
    query_conv = Conv2D(filters=f, kernel_size=1, strides=1, padding="same", name="query_conv_" + name)(inputs)
    key_conv = Conv2D(filters=f, kernel_size=1, strides=1, padding="same", name="key_conv_" + name)(inputs)
    value_conv = Conv2D(filters=in_dim, kernel_size=1, strides=1, padding="same", name="value_conv_" + name)(inputs)
    gamma = K.variable(0)

    proj_query = Reshape((height * width, f), name="proj_query_" + name)(query_conv)  # (B, H*W, C)
    proj_key = Reshape((height * width, f), name="proj_key_" + name)(key_conv)
    proj_key = Permute((2, 1), name="proj_key_transpose_" + name)(proj_key)  # (B, C, H*W)
    energy = Dot(axes=[2, 1], name="energy_" + name)([proj_query, proj_key])  # (B, H*W, H*W)
    # attention_map = Softmax(axis=-1, name="pos_att_" + name)(energy)
    attention_map = Lambda(lambda x: tf.nn.softmax(x, axis=-1), name="lambda_pos_attn_" + name)(energy)

    proj_value = Reshape((height * width, in_dim), name="proj_value_" + name)(value_conv)
    out = Dot(axes=[2, 1], name="out_" + name)([attention_map, proj_value])
    out = Reshape((height, width, in_dim), name="reshape_out_" + name)(out)
    # out_mul = Multiply()([out, gamma])
    out = Add(name="position_add_" + name)([inputs, out])
    return out


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
    print("kvar:", kvar.shape)
    # a = K.eval(kvar)
    # print(a)
    x = cbam_block_parallel(kvar, ratio=2)
    print("cbam", x)
    # print(K.eval(x))
