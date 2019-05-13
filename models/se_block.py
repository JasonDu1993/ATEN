from keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, Permute, Input
from keras import backend as K


def squeeze_excite_block(input, ratio=16):
    """ Create a squeeze-excite block
    Args:
        input: input tensor
        ratio: channel factor
    Returns:
        a keras tensor
    """
    init = input
    # print("input", input)
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    # print("se1", se)
    se = Reshape(se_shape)(se)
    # print("se2", se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    # print("se3", se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    # print("se4", se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)
        # print("se5", se)

    x = multiply([init, se])
    # print("x", x)
    return x


if __name__ == '__main__':
    a = Input(shape=(100, 200, 32))
    squeeze_excite_block(a)
