"""Keras session setup. Import this before anything that touches the GPU.

Set FOSAE_GPU=0 to hide the GPU. config_cpu.py does that for you.
"""

import os

import matplotlib
matplotlib.use('Agg')

import tensorflow as tf
import keras.backend as K

print("Default float: {}".format(K.floatx()))

# tf.compat.v1 exists in TF 1.15 and in TF 2, so this works either way.
_tf1 = tf.compat.v1 if hasattr(tf, "compat") else tf


def load_session():
    use_gpu = os.environ.get("FOSAE_GPU", "1") != "0"
    K.set_session(
        _tf1.Session(
            config=_tf1.ConfigProto(
                allow_soft_placement=True,
                intra_op_parallelism_threads=1,
                inter_op_parallelism_threads=1,
                device_count={'CPU': 1, 'GPU': 1 if use_gpu else 0},
                gpu_options=_tf1.GPUOptions(
                    per_process_gpu_memory_fraction=1.0,
                    allow_growth=True))))


load_session()
clear_session = K.clear_session


def reload_session():
    clear_session()
    load_session()
