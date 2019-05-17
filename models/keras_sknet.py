# -*- coding: utf-8 -*-
# @Time    : 2019/5/17 14:51
# @Author  : Jason
# @Email   : 1358681631@qq.com
# @File    : keras_sknet.py
# @Software: PyCharm
from keras.layers import Conv2D, Dense, Input, BatchNormalization


def SKConv(input, M, r, L=32, stride=1, is_training=True):
    input_feature = input.get_shape().as_list()[3]
    d = max(int(input_feature / r), L)
    net = input
    with slim.arg_scope([slim.conv2d, slim.fully_connected], activation_fn=tf.nn.relu):
        for i in range(M):
            net = slim.conv2d(net, input_feature, [3 + i * 2, 3 + i * 2], rate=1 + i, stride=stride)
            net = slim.batch_norm(net, decay=0.9, center=True, scale=True, epsilon=1e-5,
                                  updates_collections=tf.GraphKeys.UPDATE_OPS, is_training=is_training)
            net = tf.nn.relu(net)
            if i == 0:
                fea_U = net
            else:
                fea_U = tf.add(fea_U, net)
        gap = tflearn.global_avg_pool(net)
        fc = slim.fully_connected(gap, d, activation_fn=None)
        fcs = fc
        for _ in range(M):
            fcs = slim.fully_connected(fcs, input_feature, activation_fn=None)
            if _ == 0:
                att_vec = fcs
            else:
                att_vec = tf.add(att_vec, fcs)
        att_vec = tf.expand_dims(att_vec, axis=1)
        att_vec = tf.expand_dims(att_vec, axis=1)
        att_vec_softmax = tf.nn.softmax(att_vec)
        fea_v = tf.multiply(fea_U, att_vec_softmax)
    return fea_v


def SKConv_keras(input, M, r, L=32, stride=1, is_training=True):
    """

    Args:
        input:
        M: the number of branchs.
        r: the radio for compute d, the length of z.
        L: the minimum dim of the vector z in paper, default 32.
        stride: stride, default 1.
        is_training:
 M:
            G: num of convolution groups.
            r:
            stride:
            L:
    Returns:

    """
    input_feature = input.get_shape().as_list()[3]
    d = max(int(input_feature / r), L)
    net = input
    with slim.arg_scope([slim.conv2d, slim.fully_connected], activation_fn=tf.nn.relu):
        for i in range(M):
            net = slim.conv2d(net, input_feature, [3 + i * 2, 3 + i * 2], rate=1 + i, stride=stride)
            net = slim.batch_norm(net, decay=0.9, center=True, scale=True, epsilon=1e-5,
                                  updates_collections=tf.GraphKeys.UPDATE_OPS, is_training=is_training)
            net = tf.nn.relu(net)
            if i == 0:
                fea_U = net
            else:
                fea_U = tf.add(fea_U, net)
        gap = tflearn.global_avg_pool(net)
        fc = slim.fully_connected(gap, d, activation_fn=None)
        fcs = fc
        for _ in range(M):
            fcs = slim.fully_connected(fcs, input_feature, activation_fn=None)
            if _ == 0:
                att_vec = fcs
            else:
                att_vec = tf.add(att_vec, fcs)
        att_vec = tf.expand_dims(att_vec, axis=1)
        att_vec = tf.expand_dims(att_vec, axis=1)
        att_vec_softmax = tf.nn.softmax(att_vec)
        fea_v = tf.multiply(fea_U, att_vec_softmax)
    return fea_v
