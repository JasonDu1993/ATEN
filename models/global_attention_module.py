# -*- coding: utf-8 -*-
# @Time    : 2019/11/27 18:22
# @Author  : Jason
# @Email   : 1358681631@qq.com
# @File    : non_local.py
# @Software: PyCharm
from keras.layers import Activation, Reshape, Lambda, dot, add
from keras.layers import Conv1D, Conv2D, Conv3D, Input
from keras.layers import GlobalAveragePooling2D, Dense, Permute, multiply
from keras.layers import MaxPool1D
from keras import backend as K
from keras import layers as KL


def se_block(input_feature, ratio=8):
    """Contains the implementation of Squeeze-and-Excitation(SE) block.
    As described in https://arxiv.org/abs/1709.01507.
    """

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    shape = K.int_shape(input_feature)
    channel = shape[channel_axis]
    se_feature = GlobalAveragePooling2D()(input_feature)
    se_feature = Reshape((1, 1, channel))(se_feature)
    assert K.int_shape(se_feature)[1:] == (1, 1, channel)
    se_feature = Dense(channel // ratio,
                       activation='relu',
                       kernel_initializer='he_normal',
                       use_bias=True,
                       bias_initializer='zeros')(se_feature)
    assert K.int_shape(se_feature)[1:] == (1, 1, channel // ratio)
    se_feature = Dense(channel,
                       activation='sigmoid',
                       kernel_initializer='he_normal',
                       use_bias=True,
                       bias_initializer='zeros')(se_feature)
    assert K.int_shape(se_feature)[1:] == (1, 1, channel)
    if K.image_data_format() == 'channels_first':
        se_feature = Permute((3, 1, 2))(se_feature)

    se_feature = multiply([input_feature, se_feature])
    return se_feature


def position_se_block(input_feature, ratio=8):
    """
    """

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    pse_feature = KL.Lambda(lambda x: K.mean(x, axis=channel_axis, keepdims=True))(input_feature)
    pse_feature = Conv2D(1, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal', activation="relu"
                         )(pse_feature)
    pse_feature = Conv2D(1, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal',
                         activation="sigmoid")(
        pse_feature)
    pse_feature = multiply([input_feature, pse_feature])
    return pse_feature


def position_se_block_f9(input_feature, ratio=8, f=9):
    """
    """

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    pse_feature = KL.Lambda(lambda x: K.mean(x, axis=channel_axis, keepdims=True))(input_feature)
    pse_feature = Conv2D(f, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal', activation="relu"
                         )(pse_feature)
    pse_feature = KL.Lambda(lambda x: K.mean(x, axis=channel_axis, keepdims=True))(pse_feature)
    pse_feature = Conv2D(f, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal',
                         activation="relu")(
        pse_feature)
    pse_feature = KL.Lambda(lambda x: K.mean(x, axis=channel_axis, keepdims=True))(pse_feature)
    pse_feature = KL.Activation("sigmoid")(pse_feature)
    pse_feature = multiply([input_feature, pse_feature])
    return pse_feature


def global_attention_module(input_feature, ratio=8, add_residual=False, name="base"):
    """
    """

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    shape = K.int_shape(input_feature)
    channel = shape[channel_axis]
    se_feature = GlobalAveragePooling2D()(input_feature)
    se_feature = Reshape((1, 1, channel))(se_feature)
    assert K.int_shape(se_feature)[1:] == (1, 1, channel)
    se_feature = Dense(channel // ratio,
                       activation='relu',
                       kernel_initializer='he_normal',
                       use_bias=True,
                       bias_initializer='zeros')(se_feature)
    assert K.int_shape(se_feature)[1:] == (1, 1, channel // ratio)
    se_feature = Dense(channel,
                       activation='sigmoid',
                       kernel_initializer='he_normal',
                       use_bias=True,
                       bias_initializer='zeros')(se_feature)
    assert K.int_shape(se_feature)[1:] == (1, 1, channel)
    if K.image_data_format() == 'channels_first':
        se_feature = Permute((3, 1, 2))(se_feature)
    pse_feature = KL.Lambda(lambda x: K.mean(x, axis=channel_axis, keepdims=True))(input_feature)
    pse_feature = Conv2D(1, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal', activation="relu"
                         )(pse_feature)
    pse_feature = Conv2D(1, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal',
                         activation="sigmoid")(
        pse_feature)
    gam_feature = multiply([pse_feature, se_feature])
    gam_feature = multiply([input_feature, gam_feature])
    if add_residual:
        gam_feature = add([input_feature, gam_feature])
    return gam_feature


def global_attention_module_f(input_feature, ratio=8, f=128, add_residual=False):
    """
    """

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    shape = K.int_shape(input_feature)
    channel = shape[channel_axis]
    se_feature = GlobalAveragePooling2D()(input_feature)
    se_feature = Reshape((1, 1, channel))(se_feature)
    assert K.int_shape(se_feature)[1:] == (1, 1, channel)
    se_feature = Dense(channel // ratio,
                       activation='relu',
                       kernel_initializer='he_normal',
                       use_bias=True,
                       bias_initializer='zeros')(se_feature)
    assert K.int_shape(se_feature)[1:] == (1, 1, channel // ratio)
    se_feature = Dense(channel,
                       activation='sigmoid',
                       kernel_initializer='he_normal',
                       use_bias=True,
                       bias_initializer='zeros')(se_feature)
    assert K.int_shape(se_feature)[1:] == (1, 1, channel)
    if K.image_data_format() == 'channels_first':
        se_feature = Permute((3, 1, 2))(se_feature)
    pse_feature = KL.Lambda(lambda x: K.mean(x, axis=channel_axis, keepdims=True))(input_feature)
    pse_feature = Conv2D(f, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal', activation="relu"
                         )(pse_feature)
    pse_feature = KL.Lambda(lambda x: K.mean(x, axis=channel_axis, keepdims=True))(pse_feature)
    pse_feature = Conv2D(f, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal',
                         activation="relu")(
        pse_feature)
    pse_feature = KL.Lambda(lambda x: K.mean(x, axis=channel_axis, keepdims=True))(pse_feature)
    pse_feature = KL.Activation("sigmoid")(pse_feature)
    gam_feature = multiply([pse_feature, se_feature])
    gam_feature = multiply([input_feature, gam_feature])
    if add_residual:
        gam_feature = add([input_feature, gam_feature])
    return gam_feature


def non_local_block(ip, intermediate_dim=None, compression=2,
                    mode='embedded', add_residual=True):
    """
    Adds a Non-Local block for self attention to the input tensor.
    Input tensor can be or rank 3 (temporal), 4 (spatial) or 5 (spatio-temporal).
    Arguments:
        ip: input tensor
        intermediate_dim: The dimension of the intermediate representation. Can be
            `None` or a positive integer greater than 0. If `None`, computes the
            intermediate dimension as half of the input channel dimension.
        compression: None or positive integer. Compresses the intermediate
            representation during the dot products to reduce memory consumption.
            Default is set to 2, which states halve the time/space/spatio-time
            dimension for the intermediate step. Set to 1 to prevent computation
            compression. None or 1 causes no reduction.
        mode: Mode of operation. Can be one of `embedded`, `gaussian`, `dot` or
            `concatenate`.
        add_residual: Boolean value to decide if the residual connection should be
            added or not. Default is True for ResNets, and False for Self Attention.
    Returns:
        a tensor of same shape as input
    """
    channel_dim = 1 if K.image_data_format() == 'channels_first' else -1
    ip_shape = K.int_shape(ip)

    if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
        raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')

    if compression is None:
        compression = 1

    dim1, dim2, dim3 = None, None, None

    # check rank and calculate the input shape
    if len(ip_shape) == 3:  # temporal / time series data
        rank = 3
        batchsize, dim1, channels = ip_shape

    elif len(ip_shape) == 4:  # spatial / image data
        rank = 4

        if channel_dim == 1:
            batchsize, channels, dim1, dim2 = ip_shape
        else:
            batchsize, dim1, dim2, channels = ip_shape

    elif len(ip_shape) == 5:  # spatio-temporal / Video or Voxel data
        rank = 5

        if channel_dim == 1:
            batchsize, channels, dim1, dim2, dim3 = ip_shape
        else:
            batchsize, dim1, dim2, dim3, channels = ip_shape

    else:
        raise ValueError('Input dimension has to be either 3 (temporal), 4 (spatial) or 5 (spatio-temporal)')

    # verify correct intermediate dimension specified
    if intermediate_dim is None:
        intermediate_dim = channels // 2

        if intermediate_dim < 1:
            intermediate_dim = 1

    else:
        intermediate_dim = int(intermediate_dim)

        if intermediate_dim < 1:
            raise ValueError('`intermediate_dim` must be either `None` or positive integer greater than 1.')

    if mode == 'gaussian':  # Gaussian instantiation
        x1 = Reshape((-1, channels))(ip)  # xi
        x2 = Reshape((-1, channels))(ip)  # xj
        f = dot([x1, x2], axes=2)
        f = Activation('softmax')(f)

    elif mode == 'dot':  # Dot instantiation
        # theta path
        theta = _convND(ip, rank, intermediate_dim)
        theta = Reshape((-1, intermediate_dim))(theta)

        # phi path
        phi = _convND(ip, rank, intermediate_dim)
        phi = Reshape((-1, intermediate_dim))(phi)

        f = dot([theta, phi], axes=2)

        size = K.int_shape(f)

        # scale the values to make it size invariant
        f = Lambda(lambda z: (1. / float(size[-1])) * z)(f)

    elif mode == 'concatenate':  # Concatenation instantiation
        raise NotImplementedError('Concatenate model has not been implemented yet')

    else:  # Embedded Gaussian instantiation
        # theta path
        theta = _convND(ip, rank, intermediate_dim)
        theta = Reshape((-1, intermediate_dim))(theta)  # [B, N, intermediate_dim]

        # phi path
        phi = _convND(ip, rank, intermediate_dim)
        phi = Reshape((-1, intermediate_dim))(phi)  # phi:[B, N, intermediate_dim]

        if compression > 1:
            # shielded computation
            phi = MaxPool1D(compression)(phi)  # phi:[B, N/2, intermediate_dim]

        f = dot([theta, phi], axes=2)  # theta:[B, N, intermediate_dim], phi:[B, N/2, intermediate_dim], f:[B, N, N/2]
        f = Activation('softmax')(f)

    # g path
    g = _convND(ip, rank, intermediate_dim)
    g = Reshape((-1, intermediate_dim))(g)  # g:[B, N, intermediate_dim]

    if compression > 1 and mode == 'embedded':
        # shielded computation
        g = MaxPool1D(compression)(g)  # g:[B, N/2, intermediate_dim]

    # compute output path
    y = dot([f, g], axes=[2, 1])  # f:[B, N, N/2], g:[B, N/2, intermediate_dim], y:[B, N, intermediate_dim]

    # reshape to input tensor format
    if rank == 3:
        y = Reshape((dim1, intermediate_dim))(y)
    elif rank == 4:
        if channel_dim == -1:
            y = Reshape((dim1, dim2, intermediate_dim))(y)
        else:
            y = Reshape((intermediate_dim, dim1, dim2))(y)
    else:
        if channel_dim == -1:
            y = Reshape((dim1, dim2, dim3, intermediate_dim))(y)
        else:
            y = Reshape((intermediate_dim, dim1, dim2, dim3))(y)

    # project filters
    y = _convND(y, rank, channels)

    # residual connection
    if add_residual:
        y = add([ip, y])

    return y


def _convND(ip, rank, channels):
    assert rank in [3, 4, 5], "Rank of input must be 3, 4 or 5"

    if rank == 3:
        x = Conv1D(channels, 1, padding='same', use_bias=False, kernel_initializer='he_normal')(ip)
    elif rank == 4:
        x = Conv2D(channels, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(ip)
    else:
        x = Conv3D(channels, (1, 1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(ip)
    return x


if __name__ == '__main__':
    a = Input(shape=(100, 200, 32))
    # b = non_local_block(a)
    b = global_attention_module(a)
    print(b)
