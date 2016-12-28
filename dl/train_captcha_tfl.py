# -*- coding: utf-8 -*-

import os
from io import BytesIO
import glob
import math
import random
import sys
import PIL
import numpy as np
from numpy import argmax, array
import tflearn
from tflearn.layers.core import input_data, fully_connected, dropout, flatten
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical
from captcha.image import ImageCaptcha,WheezyCaptcha

FONTS = glob.glob('/usr/share/fonts/truetype/dejavu/*.ttf')
SAMPLE_SIZE = 20
TEST_SAMPLE_RATE = 0.3
NB_BATCH = 10
SHOW_SAMPLE_SIZE = 5
INVALID_DIGIT = -1
DIGIT_COUNT = 4
DIGIT_FORMAT_STR = "%%0%dd" % DIGIT_COUNT
CLASS_COUNT = 10
RGB_COLOR_COUNT = 3
# standard width for the whole captcha image
IMAGE_STD_WIDTH = 200
# standard height for the whole captcha image
IMAGE_STD_HEIGHT = 200
CONV1_NB_FILTERS = IMAGE_STD_HEIGHT / 2 + 2
CONV2_NB_FILTERS = IMAGE_STD_HEIGHT + 2 * 2
OUT_PUT_NAME_FORMAT = 'out_%02d'
NB_EPOCH = 10
BATCH_SIZE = 128

def generate_image_sets_for_single_digit(nb_sample=SAMPLE_SIZE, single_digit_index=0):
    captcha = ImageCaptcha()

    labels = []
    images = []
    for i in range(0, nb_sample):
        digits = 0
        last_digit = INVALID_DIGIT
        for j in range(0, DIGIT_COUNT):
            digit = last_digit
            while digit == last_digit:
                digit = random.randint(0, 9)
            last_digit = digit
            digits = digits * 10 + digit
        digits_as_str = DIGIT_FORMAT_STR % digits
        labels.append(digits_as_str)
        images.append(captcha.generate_image(digits_as_str))

    digit_labels = list()

    for digit_index in range(0, DIGIT_COUNT):
        digit_labels.append(np.empty(nb_sample, dtype="int8"))

    shape = (nb_sample, IMAGE_STD_HEIGHT, IMAGE_STD_WIDTH, RGB_COLOR_COUNT)
    digit_image_data = np.empty(shape, dtype="float32")

    for index in range(0, nb_sample):
        img = images[index].resize((IMAGE_STD_WIDTH, IMAGE_STD_HEIGHT), PIL.Image.LANCZOS)
        img_arr = np.asarray(img, dtype="float32") / 255.0

        digit_image_data[index, :, :, :] = img_arr

        for digit_index in range(0, DIGIT_COUNT):
            digit_labels[digit_index][index] = labels[index][digit_index]

    x = digit_image_data
    y = to_categorical(digit_labels[single_digit_index], CLASS_COUNT)

    return x, y

# def generate_image_sets_for_multi_digits(nb_sample=SAMPLE_SIZE):
#     captcha = ImageCaptcha()
#
#     labels = []
#     images = []
#     for i in range(0, nb_sample):
#         digits = 0
#         last_digit = INVALID_DIGIT
#         for j in range(0, DIGIT_COUNT):
#             digit = last_digit
#             while digit == last_digit:
#                 digit = random.randint(0, 9)
#             last_digit = digit
#             digits = digits * 10 + digit
#         digits_as_str = DIGIT_FORMAT_STR % digits
#         labels.append(digits_as_str)
#         images.append(captcha.generate_image(digits_as_str))
#
#     digit_labels = np.empty((nb_sample, DIGIT_COUNT), dtype="int8")
#
#     shape = (nb_sample, RGB_COLOR_COUNT, IMAGE_STD_HEIGHT, IMAGE_STD_WIDTH) if IS_TH else (nb_sample, IMAGE_STD_HEIGHT, IMAGE_STD_WIDTH, RGB_COLOR_COUNT)
#     digit_image_data = np.empty(shape, dtype="float32")
#
#     for index in range(0, nb_sample):
#         img = images[index].resize((IMAGE_STD_WIDTH, IMAGE_STD_HEIGHT), PIL.Image.LANCZOS)
#         img_arr = np.asarray(img, dtype="float32") / 255.0
#
#         digit_image_data[index, :, :, :] = img_arr
#
#         for digit_index in range(0, DIGIT_COUNT):
#             digit_labels[index][digit_index] = labels[index][digit_index]
#     x, y_as_num = digit_image_data, np.rollaxis(digit_labels, 1)
#     y = { (OUT_PUT_NAME_FORMAT % i ): np_utils.to_categorical(y_as_num[i], CLASS_COUNT) for i in range(0, DIGIT_COUNT) }
#
#     return x, y

def create_cnn_layers():
    shape = [None, IMAGE_STD_HEIGHT, IMAGE_STD_WIDTH, RGB_COLOR_COUNT]

    # input_layer = Input(name='input', shape=shape)
    input_layer = input_data(name='input', shape=shape)
    # h = Convolution2D(22, 5, 5, activation='relu', dim_ordering=dim_ordering)(input_layer)
    h = conv_2d(input_layer, 22, [5, 5], activation='relu', regularizer="L2")
    POOL_SIZE = [2, 2]
    # h = MaxPooling2D(pool_size=POOL_SIZE)(h)
    h = max_pool_2d(h, POOL_SIZE)
    # h = Convolution2D(44, 3, 3, activation='relu', dim_ordering=dim_ordering)(h)
    h = conv_2d(h, 44, [3, 3], activation='relu', regularizer="L2")
    # h = MaxPooling2D(pool_size=POOL_SIZE)(h)
    h = max_pool_2d(h, POOL_SIZE)
    # h = Dropout(0.25)(h)
    h = dropout(h, 1-0.25)
    # last_cnn_layer = Flatten()(h)
    last_cnn_layer = flatten(h)
    return input_layer, last_cnn_layer

def create_single_digit_model():
    input_layer, last_cnn_layer = create_cnn_layers()

    # h = Dense(256, activation='relu')(last_cnn_layer)
    h = fully_connected(last_cnn_layer, 256, activation='relu')
    # h = Dropout(0.5)(h)
    h = dropout(h, 1-0.5)
    # output_layer = Dense(CLASS_COUNT, activation='softmax', name='out')(h)
    output_layer = fully_connected(h, CLASS_COUNT, activation='softmax')
    network = regression(output_layer, optimizer='adadelta', learning_rate=0.01,
                     loss='categorical_crossentropy', name='out')
    # model = Model(input_layer, output_layer)
    model = tflearn.DNN(network, tensorboard_verbose=3, tensorboard_dir='./logs/')
    return model

# def create_multi_digit_model(model_file='', digit_count=DIGIT_COUNT):
#     input_layer, last_cnn_layer = create_cnn_layers()
#
#     outputs = []
#     loss = {}
#     for index in range(0, digit_count):
#         h = Dense(256, activation='relu')(last_cnn_layer)
#         h = Dropout(0.5)(h)
#         out_name = OUT_PUT_NAME_FORMAT % index
#         output = Dense(CLASS_COUNT, activation='softmax', name=out_name)(h)
#         loss[out_name] = 'categorical_crossentropy'
#         outputs.append(output)
#
#     model = Model(input=input_layer, output=outputs)
#     model.compile(
#         optimizer='adadelta',
#         loss=loss
#     )
#     return model

def print_acc(acc):
    print 'Single picture test accuracy: %2.2f%%' % (acc * 100)
    print 'Theoretical accuracy: %2.2f%% ~  %2.2f%%' % ((5*acc-4)*100, pow(acc, 5)*100)

def save_model(model, save_model_file):
    print '... saving to %s' % save_model_file
    model.save(save_model_file)

BANNER_BAR = '-----------------------------------'
BANNER_BAR = '———————————————————————————————————'

def train_single_digit_model(model, index):
    save_model_file = 'model/tflm_one_%d.hdf5' % (index + 1)
    train_sample_size = SAMPLE_SIZE
    test_sample_size = int(SAMPLE_SIZE * TEST_SAMPLE_RATE)

    def gen(nb_sample):
        x, y = generate_image_sets_for_single_digit(nb_sample)
        while True:
            yield {'input': x}, {'out': y}

    # class ValidateAcc(Callback):
    #     def on_epoch_end(self, epoch, logs={}):
    #         print
    #         print BANNER_BAR
    #         print 'Testing on %d samples...' % test_sample_size
    #         x_test, y_test_as_map = gen(test_sample_size).next()
    #         y_test = y_test_as_map['out']
    #         y_test_as_num = array([argmax(i) for i in y_test])
    #         r = model.predict(x_test, verbose=0)
    #         y_predict_as_num = array([argmax(i) for i in r])
    #         acc = sum(y_predict_as_num == y_test_as_num) / test_sample_size
    #         print_acc(acc)
    #         print BANNER_BAR

    # check_point = ModelCheckpoint(filepath=save_model_file)
    # validation = ValidateAcc()
    #
    # callbacks = [check_point, validation]
    # if not IS_TH:
    #     tb = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=False)
    #     callbacks.append(tb)

    print 'Training on %d samples...' % (train_sample_size)

    x_train, y_train = gen(test_sample_size).next()

    model.fit(
        x_train, y_train,
        NB_EPOCH,
        show_metric=True,
        run_id='tfl'
    )

    save_model(model, save_model_file)

# def train_multi_digit_model(model, index):
#     save_model_file = 'model/model_mul_%d.hdf5' % (index + 1)
#     train_sample_size = SAMPLE_SIZE
#     test_sample_size = int(SAMPLE_SIZE * TEST_SAMPLE_RATE)
#
#     def gen(nb_sample):
#         x, y = generate_image_sets_for_multi_digits(nb_sample)
#         while True:
#             yield {'input': x}, y
#
#     class ValidateAcc(Callback):
#         def on_epoch_end(self, epoch, logs={}):
#             print
#             print BANNER_BAR
#             print 'Testing on %d samples...' % test_sample_size
#
#             x_test, y_test_as_map = gen(test_sample_size).next()
#             r = model.predict(x_test, verbose=0)
#             # print r
#             # https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html
#             is_predict_correct = np.ones(test_sample_size, dtype="bool")
#             # print r[0]
#             # print r[0][0]
#             # print argmax(r[0][0])
#             # print array([argmax(j) for j in r[0]])
#             for i in range(0, DIGIT_COUNT):
#                 output_name_i = OUT_PUT_NAME_FORMAT % i
#                 # print r[i]
#                 y_predict_as_num_i = array([argmax(j) for j in r[i]])
#                 y_test_i = y_test_as_map[output_name_i]
#                 y_test_as_num_i = array([argmax(i) for i in y_test_i])
#                 # print y_predict_as_num_i, y_test_as_num_i
#                 is_predict_correct_i = y_predict_as_num_i == y_test_as_num_i
#                 acc = sum(is_predict_correct_i) / (test_sample_size * 1.0)
#                 print '[%s]:' % output_name_i
#                 print_acc(acc)
#                 is_predict_correct = is_predict_correct & is_predict_correct_i
#             acc_all = sum(is_predict_correct) / (test_sample_size * 1.0)
#             # print y_predict_i, y_test_i, test_sample_size, acc
#             print '[out]:'
#             print_acc(acc_all)
#             print BANNER_BAR
#
#     check_point = ModelCheckpoint(filepath=save_model_file) # "tmp/mul.weights.{epoch:02d}.hdf5"
#     validation = ValidateAcc()
#
#     callbacks = [check_point, validation]
#     if not IS_TH:
#         tb = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=False)
#         callbacks.append(tb)
#
#     print 'Training on %d samples...' % (train_sample_size)
#
#     model.fit_generator(
#         gen(train_sample_size / NB_BATCH),
#         samples_per_epoch=train_sample_size,
#         # batch_size=BATCH_SIZE,
#         nb_epoch=NB_EPOCH,
#         callbacks=callbacks
#     )
#
#     save_model(model, save_model_file)

print sys.argv

ARG_MODEL_TYPE = 1

MODEL_TYPE_SINGLE = 'one'
MODEL_TYPE_MULTIPLE = 'mul'

ARG_MODEL_INDEX = 2
ARG_MODEL_INDEX_MAX = 3

if len(sys.argv) > ARG_MODEL_TYPE:
    model_type = MODEL_TYPE_MULTIPLE if sys.argv[ARG_MODEL_TYPE] == MODEL_TYPE_MULTIPLE else MODEL_TYPE_SINGLE

if len(sys.argv) > ARG_MODEL_INDEX:
    index = int(sys.argv[ARG_MODEL_INDEX])
else:
    index = -1

if len(sys.argv) > ARG_MODEL_INDEX_MAX:
    max = int(sys.argv[ARG_MODEL_INDEX_MAX])
else:
    max = 1

base_model_file = '' # 'model/model_one_107.hdf5'
digit_count = DIGIT_COUNT

if model_type == MODEL_TYPE_SINGLE:
    model = create_single_digit_model()

    if index >= 0:
        model_file = 'model/model_one_%d.hdf5' % index
        print "Training based on %s" % model_file
        model.load(model_file)

    while index < max:
        train_single_digit_model(model, index)
        index = index + 1
# elif model_type == MODEL_TYPE_MULTIPLE:
#     model = create_multi_digit_model(base_model_file, digit_count)
#
#     if index >= 0:
#         model_file = 'model/model_mul_%d.hdf5' % index
#         print "Training based on %s" % model_file
#         model.load_weights(model_file)
#
#     while index < max:
#         train_multi_digit_model(model, index)
#         index = index + 1
# else:
#     pass

# for index in range(SHOW_SAMPLE_SIZE):
#     display.display(labels[index])
#     display.display(images[index])