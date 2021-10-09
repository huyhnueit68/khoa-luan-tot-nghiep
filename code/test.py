# import library necessary
import os

import cv2
import numpy as np
import pandas as pd
from keras import callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping
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

data = np.load('../data/AffectNet-8Labels/train_set/train_set/annotations/100002_aro.npy')
print(data)