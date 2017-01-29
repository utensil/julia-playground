# -*- coding: utf-8 -*-

""" Convolutional Neural Network for MNIST dataset classification task.
References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.
Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
"""

from __future__ import division, print_function, absolute_import

import numpy as np
import tensorflow as tf

learn = tf.contrib.learn
tf.logging.set_verbosity(tf.logging.ERROR)

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

# Data loading and preprocessing
import tflearn.datasets.mnist as mnist
X, Y, testX, testY = mnist.load_data(one_hot=True)

max_examples = 10000
X = X.reshape([-1, 28, 28, 1])[:max_examples]
testX = testX.reshape([-1, 28, 28, 1])

from tflearn.layers.core import flatten

IMAGE_STD_HEIGHT = 28
IMAGE_STD_WIDTH = 28
RGB_COLOR_COUNT = 1
OPTIMIZER = tflearn.optimizers.AdaDelta(learning_rate=1.0, rho=0.95)
# , epsilon=1e-08, use_locking=False, name='AdaDelta')# 'adam' # 'adadelta'
# (lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)

# This is the same as keras default glorot_normal
INIT = tflearn.initializations.xavier(uniform=False) # , seed=None, dtype=tf.float32)
CLASS_COUNT = 10

def conv_2d_specialized(incoming, nb_filter, filter_size):
    return conv_2d(incoming, nb_filter, filter_size,
        padding='valid',
        activation='relu',
        weights_init=INIT) #, regularizer="L2")

def create_cnn_layers():
    shape = [None, IMAGE_STD_HEIGHT, IMAGE_STD_WIDTH, RGB_COLOR_COUNT]

    # input_layer = Input(name='input', shape=shape)
    input_layer = input_data(name='input', shape=shape)
    # h = Convolution2D(22, 5, 5, activation='relu', dim_ordering=dim_ordering)(input_layer)
    h = conv_2d_specialized(input_layer, 22, [5, 5])
    POOL_SIZE = [2, 2]
    # h = MaxPooling2D(pool_size=POOL_SIZE)(h)
    h = max_pool_2d(h, POOL_SIZE, padding='valid')
    h = local_response_normalization(h)
    # h = Convolution2D(44, 3, 3, activation='relu', dim_ordering=dim_ordering)(h)
    h = conv_2d_specialized(h, 44, [3, 3])
    # h = MaxPooling2D(pool_size=POOL_SIZE)(h)
    h = max_pool_2d(h, POOL_SIZE, padding='valid')
    h = local_response_normalization(h)
    # h = Dropout(0.25)(h)
    h = dropout(h, 1-0.25)
    # last_cnn_layer = Flatten()(h)
    last_cnn_layer = flatten(h)
    return input_layer, last_cnn_layer

def create_single_digit_model():
    input_layer, last_cnn_layer = create_cnn_layers()

    # h = Dense(256, activation='relu')(last_cnn_layer)
    h = fully_connected(last_cnn_layer, 256, activation='relu', weights_init=INIT)
    # h = Dropout(0.5)(h)
    h = dropout(h, 1-0.5)
    # output_layer = Dense(CLASS_COUNT, activation='softmax', name='out')(h)
    output_layer = fully_connected(h, CLASS_COUNT, activation='softmax', weights_init=INIT)
    network = regression(output_layer, optimizer=OPTIMIZER,
                     learning_rate=0.1,
                     loss='categorical_crossentropy', name='out')
    # model = Model(input_layer, output_layer)
    model = tflearn.DNN(network, tensorboard_verbose=3, tensorboard_dir='./logs/')
    return model

model = create_single_digit_model()
tf.get_variable_scope().reuse_variables()
model = create_single_digit_model()
# model.fit({'input': X}, {'out': Y}, n_epoch=1,
#            validation_set=({'input': testX}, {'out': testY}),
#            snapshot_step=100, show_metric=True, run_id='conv4captcha_mnist')
