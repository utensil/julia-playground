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

FONTS = glob.glob('/usr/share/fonts/truetype/dejavu/*.ttf')
SAMPLE_SIZE = 1000
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

def generate_image_sets():
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

    X, Y_all = digit_image_data, digit_labels[0]
    X_train, X_test, y_train_num, y_test = train_test_split(X, Y_all, test_size=0.1, random_state=0)
    y_train = np_utils.to_categorical(y_train_num, CLASS_COUNT)

    return (X_train, y_train, X_test, y_test)

def create_model():
    input_layer = Input(name='input', shape=(RGB_COLOR_COUNT, IMAGE_STD_HEIGHT, IMAGE_STD_WIDTH))
    h = Convolution2D(22, 5, 5, activation='relu', dim_ordering='th')(input_layer)
    h = MaxPooling2D(pool_size=POOL_SIZE)(h)
    h = Convolution2D(44, 3, 3, activation='relu', dim_ordering='th')(h)
    h = MaxPooling2D(pool_size=POOL_SIZE)(h)
    h = Dropout(0.25)(h)
    h = Flatten()(h)
    h = Dense(256, activation='relu')(h)
    h = Dropout(0.5)(h)
    output_layer = Dense(CLASS_COUNT, activation='softmax', name='out')(h)

    model = Model(input_layer, output_layer)

    # graph = Graph()
    # # graph.add_input(name='input', input_shape=(3, 40, 40))
    # graph.add_input(name='input', input_shape=(RGB_COLOR_COUNT, IMAGE_STD_HEIGHT, IMAGE_STD_WIDTH))
    # # http://stackoverflow.com/questions/36243536/what-is-the-number-of-filter-in-cnn/36243662
    # graph.add_node(Convolution2D(22, 5, 5, activation='relu', dim_ordering='th'), name='conv1', input='input')
    # graph.add_node(MaxPooling2D(pool_size=POOL_SIZE), name='pool1', input='conv1')
    # graph.add_node(Convolution2D(44, 3, 3, activation='relu', dim_ordering='th'), name='conv2', input='pool1')
    # graph.add_node(MaxPooling2D(pool_size=POOL_SIZE), name='pool2', input='conv2')
    # graph.add_node(Dropout(0.25), name='drop', input='pool2')
    # graph.add_node(Flatten(), name='flatten', input='drop')
    # graph.add_node(Dense(256, activation='relu'), name='ip', input='flatten')
    # graph.add_node(Dropout(0.5), name='drop_out', input='ip')
    # graph.add_node(Dense(CLASS_COUNT, activation='softmax'), name='result', input='drop_out')
    # graph.add_output(name='out', input='result')

    model.compile(
        optimizer='adadelta',
        loss={
            'out': 'categorical_crossentropy',
        }
    )

    return model

def train(model, X_train, y_train, X_test, y_test, index):
    class ValidateAcc(Callback):
        def on_epoch_end(self, epoch, logs={}):
            print '\n————————————————————————————————————'
            # model.load_weights('tmp/weights.%02d.hdf5' % epoch)
            r = model.predict(X_test, verbose=0)
            y_predict = array([argmax(i) for i in r])
            length = len(y_predict) * 1.0
            acc = sum(y_predict == y_test) / length
            print 'Single picture test accuracy: %2.2f%%' % (acc * 100)
            print 'Theoretical accuracy: %2.2f%% ~  %2.2f%%' % ((5*acc-4)*100, pow(acc, 5)*100)
            print '————————————————————————————————————'

    check_point = ModelCheckpoint(filepath="tmp/weights.{epoch:02d}.hdf5")
    back = ValidateAcc()
    print 'Begin train on %d samples... test on %d samples...' % (len(y_train), len(y_test))
    if index >= 0:
        model_file = 'model/model_2_%d.hdf5' % index
        print "Training based on %s" % model_file
        model.load_weights(model_file)
    model.fit(
        {'input': X_train}, {'out': y_train},
        batch_size=128, nb_epoch=3, callbacks=[check_point, back]
    )
    print '... saving'
    save_model_file = 'model/model_2_%d.hdf5' % (index + 1)
    model.save_weights(save_model_file, overwrite=True)

# print sys.argv[1]
if len(sys.argv) > 1:
    index = int(sys.argv[1])
else:
    index = -1

if len(sys.argv) > 2:
    max = int(sys.argv[2])
else:
    max = 10

model = create_model()

while index < max:
    X_train, y_train, X_test, y_test = generate_image_sets()
    train(model, X_train, y_train, X_test, y_test, index)
    index = index + 1

# for index in range(SHOW_SAMPLE_SIZE):
#     display.display(labels[index])
#     display.display(images[index])
