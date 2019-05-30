# -*- coding: utf-8 -*-
# @Time    : 2019/5/30 11:21
# @Author  : Jason
# @Email   : 1358681631@qq.com
# @File    : get_model_flops.py
# @Software: PyCharm
import tensorflow as tf
import keras.backend as K


def get_flops(model):
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()

    # We use the Keras session graph in the call to the profiler.
    flops = tf.profiler.profile(graph=K.get_session().graph,
                                run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops  # Prints the "flops" of the model.


# .... Define your model here ....
# print(get_flops(model))
