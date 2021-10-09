# import library necessary
import os

import cv2
import numpy as np
import pandas as pd
from keras import callbacks
from keras.layers import Dense, Activation, Dropout, Flatten, Concatenate
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.losses import categorical_crossentropy
from keras.models import Sequential, Model
from keras.optimizer_v1 import Adagrad
from keras.optimizers import Adam
from keras.utils import np_utils
from keras_preprocessing.image import ImageDataGenerator
from numpy import concatenate
from pylab import rcParams
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.layers import BatchNormalization

# Function init model training
def initModelTraining(input__shape, num_labels):
    # call model Sequential
    model__fer = Sequential()

    # create a Sequential model incrementally
    # add convolution 2D
    model__fer.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input__shape))
    model__fer.add(BatchNormalization())
    model__fer.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
    model__fer.add(BatchNormalization())
    model__fer.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model__fer.add(Dropout(0.5))

    # 2nd convolution layer, 2D convolution layer (spatial convolution over images)
    model__fer.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model__fer.add(BatchNormalization())
    model__fer.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model__fer.add(BatchNormalization())
    model__fer.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model__fer.add(Dropout(0.5))

    # 3rd convolution layer
    model__fer.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model__fer.add(BatchNormalization())
    model__fer.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model__fer.add(BatchNormalization())
    model__fer.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model__fer.add(Flatten())  # Return a copy of the array collapsed into one dimension.

    # fully connected neural networks
    model__fer.add(Dense(1024, activation='relu'))  # Activate linear unit Rectified
    model__fer.add(Dropout(0.2))
    model__fer.add(Dense(1024, activation='relu'))
    model__fer.add(Dropout(0.2))

    # evaluate the categorical probabilities of the input data by softmax
    model__fer.add(Dense(num_labels, activation='softmax'))

    # Compliling the model
    model__fer.compile(loss=categorical_crossentropy,  # Computes the categorical crossentropy loss
                       optimizer=Adam(),  # Optimizer that implements the Adam algorithm
                       metrics=['accuracy'])  # judge the performance of your model
    return model__fer


# init size image and number of filters
num_features = 64
num_labels = 9
batch_size = 64
num_epochs = 30
width, height = 48, 48
















