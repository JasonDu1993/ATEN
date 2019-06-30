# -*- coding: utf-8 -*-
# @Time    : 2019/6/13 10:39
# @Author  : Jason
# @Email   : 1358681631@qq.com
# @File    : DANet.py
# @Software: PyCharm
import numpy as np
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, Multiply
from keras.layers import Permute, Concatenate, Conv2D, Add, Activation, Lambda, Input, Dot
import keras.backend as K


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
    import tensorflow as tf
    attention_map = tf.nn.softmax(energy, axis=-1)

    proj_value = Reshape((height * width, in_dim), name="proj_value_" + name)(value_conv)
    out = Dot(axes=[2, 1], name="out_" + name)([attention_map, proj_value])
    out = Reshape((height, width, in_dim), name="reshape_out_" + name)(out)
    # out_mul = Multiply()([out, gamma])
    out = Add(name="position_add_"+name)([inputs, out])
    return out


def channel_attention_module(inputs, name, ratio=8):
    """ Channel attention module"""
    # Ref from SAGAN
    m_batchsize, height, width, in_dim = K.int_shape(inputs)
    # f = in_dim // ratio
    query_conv = Conv2D(filters=in_dim, kernel_size=1, strides=1, padding="same", name="query_conv_" + name)(inputs)
    key_conv = Conv2D(filters=in_dim, kernel_size=1, strides=1, padding="same", name="key_conv_" + name)(inputs)
    value_conv = Conv2D(filters=in_dim, kernel_size=1, strides=1, padding="same", name="value_conv_" + name)(inputs)
    gamma = K.variable(0)

    proj_query = Reshape((height * width, in_dim), name="proj_query_" + name)(query_conv)  # (B, H*W, C)
    proj_query = Permute((2, 1), name="proj_query_transpose_" + name)(proj_query)  # (B, C, H*W)
    proj_key = Reshape((height * width, in_dim), name="proj_key_" + name)(key_conv)  # (B, H*W, C)

    energy = Dot(axes=[2, 1], name="energy_" + name)([proj_query, proj_key])  # (B, C, C)
    attention_map = Softmax(axis=-1, name="posistion_attention_" + name)(energy)

    proj_value = Reshape((height * width, in_dim), name="proj_value_" + name)(value_conv)
    out = Dot(axes=[2, 1], name="out_" + name)([proj_value, attention_map])
    out = Reshape((height, width, in_dim), name="reshape_out_" + name)(out)
    out_mul = Multiply()([gamma, out])
    out = Add()([inputs, out_mul])
    return out


def cam(inputs, name):
    m_batchsize, height, width, C = K.int_shape(inputs)
    proj_query = Reshape((height * width, C), name="proj_query_" + name)(inputs)  # (B, H*W, C)
    proj_query = Permute((2, 1), name="proj_key_transpose_" + name)(proj_query)  # (B, C, H*W)
    proj_key = Reshape((height * width, C), name="proj_key_" + name)(inputs)  # (B, H*W, C)
    energy = Dot(axes=[2, 1], name="energy_" + name)([proj_query, proj_key])  # (B, C, C)
    energy_new = K.max(energy, axis=-1, keepdims=True) - energy
    attention_map = Softmax(axis=-1, name="channel_attention_" + name)(energy_new)

    proj_value = Reshape((height * width, C), name="proj_value_" + name)(inputs)
    out = Dot(axes=[2, 1], name="energy_" + name)([proj_value, attention_map])  # (B, H*W, C)
    out = Reshape((height, width, C), name="proj_value_" + name)(out)

    # gamma = K.variable(0)
    # out = gamma * out + inputs
    out = Add(name="channel_add_"+name)([inputs, out])
    return out


def DAModule(inputs, name, ratio=8):
    pa_out = position_attention_module(inputs, name="pam_" + name, ratio=8)
    dam_out = pa_out
    # ca_out = cam(inputs, name="cam_" + name)
    # w1 = K.variable(0)
    # w2 = K.variable(0)
    # dam_out = inputs + w1 * pa_out + w2 * ca_out
    # dam_out = Add(name="add_"+name)([pa_out, ca_out])
    return dam_out


if __name__ == '__main__':
    kvar = K.reshape(K.variable(np.arange(24), dtype="float32"), shape=(1, 4, 3, 2))  # 默认dtype="float32"
    # kvar = Input(shape=(64, 64, 128))
    # out = channel_attention_module(kvar, "pam", ratio=2)
    # out = cam(kvar, "cam")
    out = DAModule(kvar, name="dam")
    print(out)
    print(K.eval(kvar))
    print("....")
    # print(K.eval(q), q.shape)
    print(".....")
    print(K.eval(out), out.shape)
