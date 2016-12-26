# -*- coding: utf-8 -*-

from io import BytesIO
import glob
import math
import random
import sys
import PIL
import numpy as np
from numpy import argmax, array
from sklearn.cross_validation import train_test_split
from keras.callbacks import Callback, ModelCheckpoint
from keras.models import Model
from keras.utils import np_utils
from keras.layers import merge, Convolution2D, MaxPooling2D, Input, Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from captcha.image import ImageCaptcha,WheezyCaptcha

USE_PLOT = True
try:
    from keras.utils.visualize_util import plot as keras_plot
except:
    USE_PLOT = False
USE_PLOT = False
plot = keras_plot if USE_PLOT else (lambda model, file_name: model)

FONTS = glob.glob('/usr/share/fonts/truetype/dejavu/*.ttf')
SAMPLE_SIZE = 10
SHOW_SAMPLE_SIZE = 5
INVALID_DIGIT = -1
DIGIT_COUNT = 4
DIGIT_FORMAT_STR = "%%0%dd" % DIGIT_COUNT
CLASS_COUNT = 10
RGB_COLOR_COUNT = 3
POOL_SIZE = (2, 2)
# standard width for the whole captcha image
IMAGE_STD_WIDTH = 200
# standard height for the whole captcha image
IMAGE_STD_HEIGHT = 200
CONV1_NB_FILTERS = IMAGE_STD_HEIGHT / 2 + 2
CONV2_NB_FILTERS = IMAGE_STD_HEIGHT + 2 * 2
OUT_PUT_NAME_FORMAT = 'out_%02d'
NB_EPOCH = 10
BATCH_SIZE = 128

def generate_image_sets_for_single_digit(singl_digit_index=0):
    captcha = ImageCaptcha()

    # print DIGIT_FORMAT_STR
    labels = []
    images = []
    for i in range(0, SAMPLE_SIZE):
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
        digit_labels.append(np.empty(SAMPLE_SIZE, dtype="int8"))

    digit_image_data = np.empty((SAMPLE_SIZE, 3, IMAGE_STD_HEIGHT, IMAGE_STD_WIDTH), dtype="float32")

    for index in range(0, SAMPLE_SIZE):
        img = images[index].resize((IMAGE_STD_WIDTH, IMAGE_STD_HEIGHT), PIL.Image.LANCZOS)
        # if index < SHOW_SAMPLE_SIZE:
            # display.display(img)
        img_arr = np.asarray(img, dtype="float32") / 255.0
        digit_image_data[index, :, :, :] = np.rollaxis(img_arr, 2)
        for digit_index in range(0, DIGIT_COUNT):
            digit_labels[digit_index][index] = labels[index][digit_index]

    X, Y_all = digit_image_data, digit_labels[singl_digit_index]
    x_train, x_test, y_train_as_num, y_test_as_num = train_test_split(X, Y_all, test_size=0.1, random_state=0)
    y_train = np_utils.to_categorical(y_train_as_num, CLASS_COUNT)
    y_test = y_test_as_num

    return (x_train, y_train, x_test, y_test)

def generate_image_sets_for_multi_digits():
    captcha = ImageCaptcha()

    # print DIGIT_FORMAT_STR
    labels = []
    images = []
    for i in range(0, SAMPLE_SIZE):
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

    digit_labels = np.empty((SAMPLE_SIZE, DIGIT_COUNT), dtype="int8")

    digit_image_data = np.empty((SAMPLE_SIZE, 3, IMAGE_STD_HEIGHT, IMAGE_STD_WIDTH), dtype="float32")

    for index in range(0, SAMPLE_SIZE):
        img = images[index].resize((IMAGE_STD_WIDTH, IMAGE_STD_HEIGHT), PIL.Image.LANCZOS)
        # if index < SHOW_SAMPLE_SIZE:
            # display.display(img)
        img_arr = np.asarray(img, dtype="float32") / 255.0
        digit_image_data[index, :, :, :] = np.rollaxis(img_arr, 2)
        for digit_index in range(0, DIGIT_COUNT):
            digit_labels[index][digit_index] = labels[index][digit_index]

    X, Y_all = digit_image_data, digit_labels
    x_train, x_test, y_train_as_num, y_test_as_num = train_test_split(X, Y_all, test_size=0.1, random_state=0)
    y_train_as_num = np.rollaxis(y_train_as_num, 1)
    y_test_as_num = np.rollaxis(y_test_as_num, 1)

    y_train = { (OUT_PUT_NAME_FORMAT % i ): np_utils.to_categorical(y_train_as_num[i], CLASS_COUNT) for i in range(0, DIGIT_COUNT) }
    y_test = { (OUT_PUT_NAME_FORMAT % i ): y_test_as_num[i] for i in range(0, DIGIT_COUNT) }

    return (x_train, y_train, x_test, y_test)

def create_cnn_layers():
    input_layer = Input(name='input', shape=(RGB_COLOR_COUNT, IMAGE_STD_HEIGHT, IMAGE_STD_WIDTH))
    h = Convolution2D(22, 5, 5, activation='relu', dim_ordering='th')(input_layer)
    h = MaxPooling2D(pool_size=POOL_SIZE)(h)
    h = Convolution2D(44, 3, 3, activation='relu', dim_ordering='th')(h)
    h = MaxPooling2D(pool_size=POOL_SIZE)(h)
    h = Dropout(0.25)(h)
    last_cnn_layer = Flatten()(h)
    return (input_layer, last_cnn_layer)

def create_single_digit_model():
    input_layer, last_cnn_layer = create_cnn_layers()

    h = Dense(256, activation='relu')(last_cnn_layer)
    h = Dropout(0.5)(h)
    output_layer = Dense(CLASS_COUNT, activation='softmax', name='out')(h)

    model = Model(input_layer, output_layer)

    model.compile(
        optimizer='adadelta',
        loss={
            'out': 'categorical_crossentropy',
        }
    )

    return model

def create_multi_digit_model(model_file='', digit_count=DIGIT_COUNT):
    input_layer, last_cnn_layer = create_cnn_layers()

    outputs = []
    loss = {}
    for index in range(0, digit_count):
        h = Dense(256, activation='relu')(last_cnn_layer)
        h = Dropout(0.5)(h)
        out_name = OUT_PUT_NAME_FORMAT % index
        output = Dense(CLASS_COUNT, activation='softmax', name=out_name)(h)
        loss[out_name] = 'categorical_crossentropy'
        outputs.append(output)

    model = Model(input=input_layer, output=outputs)
    model.compile(
        optimizer='adadelta',
        loss=loss
    )
    return model

def print_acc(acc):
    print 'Single picture test accuracy: %2.2f%%' % (acc * 100)
    print 'Theoretical accuracy: %2.2f%% ~  %2.2f%%' % ((5*acc-4)*100, pow(acc, 5)*100)

def save_model(model, save_model_file):
    print '... saving to %s' % save_model_file
    model.save_weights(save_model_file, overwrite=True)

BANNER_BAR = '-----------------------------------'
BANNER_BAR = '———————————————————————————————————'

def train_single_digit_model(model, x_train, y_train, x_test, y_test, index):
    save_model_file = 'model/model_one_%d.hdf5' % (index + 1)

    class ValidateAcc(Callback):
        def on_epoch_end(self, epoch, logs={}):
            print
            print BANNER_BAR
            # model.load_weights('tmp/weights.%02d.hdf5' % epoch)
            r = model.predict(x_test, verbose=0)
            y_predict = array([argmax(i) for i in r])
            length = len(y_predict) * 1.0
            acc = sum(y_predict == y_test) / length
            print_acc(acc)
            print BANNER_BAR

    check_point = ModelCheckpoint(filepath="tmp/weights.{epoch:02d}.hdf5")
    back = ValidateAcc()
    print 'Begin train on %d samples... test on %d samples...' % (len(x_train), len(x_test))
    if index >= 0:
        model_file = 'model/model_one_%d.hdf5' % index
        print "Training based on %s" % model_file
        model.load_weights(model_file)
    model.fit(
        {'input': x_train}, {'out': y_train},
        batch_size=BATCH_SIZE, nb_epoch=NB_EPOCH, callbacks=[check_point, back]
    )
    save_model(model, save_model_file)

def train_multi_digit_model(model, x_train, y_train, x_test, y_test, index):
    save_model_file = 'model/model_mul_%d.hdf5' % (index + 1)

    class ValidateAcc(Callback):
        def on_epoch_end(self, epoch, logs={}):
            print
            print BANNER_BAR
            length = len(x_test)
            r = model.predict(x_test, verbose=0)
            # https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html
            is_predict_correct = np.ones(length, dtype="bool")
            # print r[0]
            # print r[0][0]
            # print argmax(r[0][0])
            # print array([argmax(j) for j in r[0]])
            for i in range(0, DIGIT_COUNT):
                output_name_i = OUT_PUT_NAME_FORMAT % i
                y_predict_i = array([argmax(j) for j in r[i]])
                y_test_i = y_test[output_name_i]
                is_predict_correct_i = y_predict_i == y_test_i
                acc = sum(is_predict_correct_i) / (length * 1.0)
                # print y_predict_i, y_test_i, length, acc
                print '[%s]:' % output_name_i
                print_acc(acc)
                is_predict_correct = is_predict_correct & is_predict_correct_i
            acc_all = sum(is_predict_correct) / (length * 1.0)
            # print y_predict_i, y_test_i, length, acc
            print '[out]:'
            print_acc(acc_all)
            print BANNER_BAR

    # check_point = ModelCheckpoint(filepath="tmp/mul.weights.{epoch:02d}.hdf5")
    check_point = ModelCheckpoint(filepath=save_model_file)
    back = ValidateAcc()
    print 'Begin train on %d samples... test on %d samples...' % (len(x_train), len(x_test))
    if index >= 0:
        model_file = 'model/model_mul_%d.hdf5' % index
        print "Training based on %s" % model_file
        model.load_weights(model_file)
    model.fit(
        {'input': x_train}, y_train,
        batch_size=BATCH_SIZE, nb_epoch=NB_EPOCH,
        callbacks=[check_point, back]
    )
    save_model(model, save_model_file)

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
model = create_single_digit_model()
plot(model, 'single_digit_model.png')
model = create_multi_digit_model(base_model_file, digit_count)
plot(model, '%d_digit_model.png' % digit_count)

if model_type == MODEL_TYPE_SINGLE:
    model = create_single_digit_model()

    while index < max:
        x_train, y_train, x_test, y_test = generate_image_sets_for_single_digit()
        train_single_digit_model(model, x_train, y_train, x_test, y_test, index)
        index = index + 1
elif model_type == MODEL_TYPE_MULTIPLE:
    while index < max:
        x_train, y_train, x_test, y_test = generate_image_sets_for_multi_digits()
        # print x_train, y_train, x_test, y_test
        train_multi_digit_model(model, x_train, y_train, x_test, y_test, index)
        index = index + 1
else:
    pass

# for index in range(SHOW_SAMPLE_SIZE):
#     display.display(labels[index])
#     display.display(images[index])
