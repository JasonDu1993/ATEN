import keras
import keras.backend as K
import keras.engine as KE
import keras.layers as KL
import keras.models as KM


def identity_block_share(input_tensor_list, kernel_size, filters, stage, block,
                         use_bias=True):
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

    conv1 = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                      use_bias=use_bias)
    bn1 = KL.BatchNormalization(axis=-1, name=bn_name_base + '2a')
    conv2 = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                      name=conv_name_base + '2b', use_bias=use_bias)
    bn2 = KL.BatchNormalization(axis=-1, name=bn_name_base + '2b')
    conv3 = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                      use_bias=use_bias)
    bn3 = KL.BatchNormalization(axis=-1, name=bn_name_base + '2c')

    features = []
    for input_tensor in input_tensor_list:
        x = conv1(input_tensor)
        x = bn1(x)
        x = KL.Activation('relu')(x)

        x = conv2(x)
        x = bn2(x)
        x = KL.Activation('relu')(x)

        x = conv3(x)
        x = bn3(x)

        x = KL.Add()([x, input_tensor])
        x = KL.Activation('relu')(x)
        features.append(x)
    return features


def conv_block_share(input_tensor_list, kernel_size, filters, stage, block,
                     strides=(2, 2), use_bias=True):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor_list: input tensor
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

    conv1 = KL.Conv2D(nb_filter1, (1, 1), strides=strides,
                      name=conv_name_base + '2a', use_bias=use_bias)
    bn1 = KL.BatchNormalization(axis=-1, name=bn_name_base + '2a')

    conv2 = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                      name=conv_name_base + '2b', use_bias=use_bias)
    bn2 = KL.BatchNormalization(axis=-1, name=bn_name_base + '2b')

    conv3 = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                      use_bias=use_bias)
    bn3 = KL.BatchNormalization(axis=-1, name=bn_name_base + '2c')

    conv4 = KL.Conv2D(nb_filter3, (1, 1), strides=strides,
                      name=conv_name_base + '1', use_bias=use_bias)
    bn4 = KL.BatchNormalization(axis=-1, name=bn_name_base + '1')

    features = []

    for input_tensor in input_tensor_list:
        x = conv1(input_tensor)
        x = bn1(x)
        x = KL.Activation('relu')(x)

        x = conv2(x)
        x = bn2(x)
        x = KL.Activation('relu')(x)

        x = conv3(x)
        x = bn3(x)

        shortcut = conv4(input_tensor)
        shortcut = bn4(shortcut)

        x = KL.Add()([x, shortcut])
        x = KL.Activation('relu')(x)

        features.append(x)
    return features


# Atrous-Convolution version of residual blocks
def atrous_identity_block_share(input_tensor_list, kernel_size, filters, stage,
                                block, atrous_rate=(2, 2), use_bias=True):
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

    conv1 = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                      use_bias=use_bias)
    bn1 = KL.BatchNormalization(axis=-1, name=bn_name_base + '2a')

    conv2 = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), dilation_rate=atrous_rate,
                      padding='same', name=conv_name_base + '2b', use_bias=use_bias)
    bn2 = KL.BatchNormalization(axis=-1, name=bn_name_base + '2b')

    conv3 = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                      use_bias=use_bias)
    bn3 = KL.BatchNormalization(axis=-1, name=bn_name_base + '2c')

    features = []
    for input_tensor in input_tensor_list:
        x = conv1(input_tensor)
        x = bn1(x)
        x = KL.Activation('relu')(x)

        x = conv2(x)
        x = bn2(x)
        x = KL.Activation('relu')(x)

        x = conv3(x)
        x = bn3(x)

        x = KL.Add()([x, input_tensor])
        x = KL.Activation('relu')(x)
        features.append(x)
    return features


def atrous_conv_block_share(input_tensor_list, kernel_size, filters, stage,
                            block, strides=(1, 1), atrous_rate=(2, 2), use_bias=True):
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

    conv1 = KL.Conv2D(nb_filter1, (1, 1), strides=strides,
                      name=conv_name_base + '2a', use_bias=use_bias)
    bn1 = KL.BatchNormalization(axis=-1, name=bn_name_base + '2a')

    conv2 = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', dilation_rate=atrous_rate,
                      name=conv_name_base + '2b', use_bias=use_bias)
    bn2 = KL.BatchNormalization(axis=-1, name=bn_name_base + '2b')

    conv3 = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                      use_bias=use_bias)
    bn3 = KL.BatchNormalization(axis=-1, name=bn_name_base + '2c')

    conv4 = KL.Conv2D(nb_filter3, (1, 1), strides=strides,
                      name=conv_name_base + '1', use_bias=use_bias)
    bn4 = KL.BatchNormalization(axis=-1, name=bn_name_base + '1')

    features = []
    for input_tensor in input_tensor_list:
        x = conv1(input_tensor)
        x = bn1(x)
        x = KL.Activation('relu')(x)

        x = conv2(x)
        x = bn2(x)
        x = KL.Activation('relu')(x)

        x = conv3(x)
        x = bn3(x)

        shortcut = conv4(input_tensor)
        shortcut = bn4(shortcut)

        x = KL.Add()([x, shortcut])
        x = KL.Activation('relu')(x)
        features.append(x)
    return features


def deeplab_resnet_share(img_inputs, architecture):
    """
    Build the architecture of resnet-101.
    img_inputs: a list of input image
    architecture:Str, "resnet50" or "resnet101"
    img_inputs = <class 'list'>: [<tf.Tensor 'input_image_key1:0' shape=(?, 512, 512, 3) dtype=float32>,
    <tf.Tensor 'input_image_key2:0' shape=(?, 512, 512, 3) dtype=float32>,
    <tf.Tensor 'input_image_key3:0' shape=(?, 512, 512, 3) dtype=float32>]

    c1 = <class 'list'>: [<tf.Tensor 'max_pooling2d_1/MaxPool:0' shape=(?, 128, 128, 64) dtype=float32>,
    <tf.Tensor 'max_pooling2d_2/MaxPool:0' shape=(?, 128, 128, 64) dtype=float32>,
    <tf.Tensor 'max_pooling2d_3/MaxPool:0' shape=(?, 128, 128, 64) dtype=float32>]

    c2 = <class 'list'>: [<tf.Tensor 'activation_24/Relu:0' shape=(?, 128, 128, 256) dtype=float32>,
    <tf.Tensor 'activation_27/Relu:0' shape=(?, 128, 128, 256) dtype=float32>,
     <tf.Tensor 'activation_30/Relu:0' shape=(?, 128, 128, 256) dtype=float32>]

    c3 = <class 'list'>: [<tf.Tensor 'activation_60/Relu:0' shape=(?, 64, 64, 512) dtype=float32>,
    <tf.Tensor 'activation_63/Relu:0' shape=(?, 64, 64, 512) dtype=float32>,
    <tf.Tensor 'activation_66/Relu:0' shape=(?, 64, 64, 512) dtype=float32>]

    c4 = <class 'list'>: [<tf.Tensor 'activation_114/Relu:0' shape=(?, 32, 32, 1024) dtype=float32>,
    <tf.Tensor 'activation_117/Relu:0' shape=(?, 32, 32, 1024) dtype=float32>,
    <tf.Tensor 'activation_120/Relu:0' shape=(?, 32, 32, 1024) dtype=float32>]

    c5 = <class 'list'>: [<tf.Tensor 'activation_141/Relu:0' shape=(?, 32, 32, 2048) dtype=float32>,
    <tf.Tensor 'activation_144/Relu:0' shape=(?, 32, 32, 2048) dtype=float32>,
    <tf.Tensor 'activation_147/Relu:0' shape=(?, 32, 32, 2048) dtype=float32>]
    """

    # Stage 1
    conv1 = KL.Conv2D(64, (7, 7), strides=(2, 2),
                      name='conv1', use_bias=False)
    bn_conv1 = KL.BatchNormalization(axis=-1, name='bn_conv1')
    c1 = []
    for img_input in img_inputs:
        x = KL.ZeroPadding2D((3, 3))(img_input)
        x = conv1(x)
        x = bn_conv1(x)
        x = KL.Activation('relu')(x)
        x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
        c1.append(x)

    # Stage 2
    features = conv_block_share(c1, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), use_bias=False)
    features = identity_block_share(features, 3, [64, 64, 256], stage=2, block='b', use_bias=False)
    c2 = features = identity_block_share(features, 3, [64, 64, 256], stage=2, block='c', use_bias=False)
    # Stage 3
    features = conv_block_share(features, 3, [128, 128, 512], stage=3, block='a', use_bias=False)
    features = identity_block_share(features, 3, [128, 128, 512], stage=3, block='b1', use_bias=False)
    features = identity_block_share(features, 3, [128, 128, 512], stage=3, block='b2', use_bias=False)
    c3 = features = identity_block_share(features, 3, [128, 128, 512], stage=3, block='b3', use_bias=False)
    # Stage 4
    features = conv_block_share(features, 3, [256, 256, 1024], stage=4, block='a', use_bias=False)
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(1, block_count + 1):
        features = identity_block_share(features, 3, [256, 256, 1024], stage=4, block='b%d' % i, use_bias=False)
    c4 = features
    # Stage 5
    features = atrous_conv_block_share(features, 3, [512, 512, 2048], stage=5, block='a', atrous_rate=(2, 2),
                                       use_bias=False)
    features = atrous_identity_block_share(features, 3, [512, 512, 2048], stage=5, block='b', atrous_rate=(2, 2),
                                           use_bias=False)
    c5 = atrous_identity_block_share(features, 3, [512, 512, 2048], stage=5, block='c', atrous_rate=(2, 2),
                                     use_bias=False)

    return c1, c5

