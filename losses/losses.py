import tensorflow as tf
import keras.backend as K


def mrcnn_global_parsing_loss_graph(num_classes, gt_parsing_map, predict_parsing_map):
    """
    gt_parsing_map: [batch, image_height, image_width] of uint8
    predict_parsing_map: [batch, height, width, parsing_class_num]
    """
    gt_shape = tf.shape(gt_parsing_map)
    predict_parsing_map = tf.image.resize_bilinear(predict_parsing_map, gt_shape[1:3])

    pred_shape = tf.shape(predict_parsing_map)

    raw_gt = tf.expand_dims(gt_parsing_map, -1)
    # raw_gt = tf.image.resize_nearest_neighbor(raw_gt, pred_shape[1:3])
    raw_gt = tf.reshape(raw_gt, [-1, ])
    raw_gt = tf.cast(raw_gt, tf.int32)

    raw_prediction = tf.reshape(predict_parsing_map, [-1, pred_shape[-1]])

    # Predictions: ignoring all predictions with labels greater or equal than n_classes
    indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, num_classes - 1)), 1)
    gt = tf.gather(raw_gt, indices)
    prediction = tf.gather(raw_prediction, indices)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=gt, logits=prediction)
    loss = tf.reduce_mean(loss)
    # loss = tf.reshape(loss, [1, 1])
    return K.cast(loss, dtype="float32")

