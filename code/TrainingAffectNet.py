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


def initModelTraining(input__shape, num_labels):
    # call model Sequential
    model__fer = Sequential()

    # create a Sequential model incrementally
    # add convolution 2D
    model__fer.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input__shape))
    model__fer.add(BatchNormalization())
    model__fer.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
    model__fer.add(BatchNormalization()) #zero mean
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

# TRAINING AFFECT NET DATASET

# init 4 array for training
X_train_affect, train_y_affect, X_test_affect, test_y_affect = [], [], [], []
img_list_train, img_data_train = [], []
img_list_test, img_data_test = [], []

# load all info training
df_train = pd.read_csv('../data/AffectNet/train-sample-affectnet.csv')

print("Load training csv")
for index, row in df_train.iterrows():
    try:
        # get image
        path_Image = row.image
        input_img = cv2.imread(path_Image)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        input_img_resize = cv2.resize(input_img, (48, 48))
        # convert matrix 2 dimensional to one dimensional vector
        input_img_resize = input_img_resize.flatten()
        img_list_train.append(np.array(input_img_resize, 'float32'))
        # append lable
        train_y_affect.append(int(row.emotion))
    except:
        print(f"error occured at index :{index} ")
print("Load training csv done!")

X_train_affect = np.array(img_list_train, 'float32')
train_y_affect = np.array(train_y_affect, 'float32')
train_y_affect = np_utils.to_categorical(train_y_affect, num_classes=num_labels)

# load all info valid
df_valid = pd.read_csv('../data/AffectNet/valid-sample-affectnet.csv')

print("Load valid csv")
for index, row in df_valid.iterrows():
    try:
        # get image
        path_Image = row.image
        input_img = cv2.imread(path_Image)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        input_img_resize = cv2.resize(input_img, (48, 48))
        # trải ma trận ảnh thành 1 chiều
        input_img_resize = input_img_resize.flatten()
        img_list_test.append(np.array(input_img_resize, 'float32'))
        # append lable
        test_y_affect.append(row.emotion)
    except:
        print(f"error occured at index :{index} and row:{row}")
print("Load valid csv done!")

X_test_affect = np.array(img_list_test, 'float32')
test_y_affect = np.array(test_y_affect, 'float32')
test_y_affect = np_utils.to_categorical(test_y_affect, num_classes=num_labels)

# Compute the arithmetic mean along the specified axis and Compute the standard deviation along the specified axis.
X_train_affect -= np.mean(X_train_affect, axis=0)  # normalize data between 0 and 1
X_train_affect /= np.std(X_train_affect, axis=0)

X_test_affect -= np.mean(X_test_affect, axis=0)
X_test_affect /= np.std(X_test_affect, axis=0)

# matrix transpose
X_train_affect = X_train_affect.reshape(X_train_affect.shape[0], 48, 48, 1)
X_test_affect = X_test_affect.reshape(X_test_affect.shape[0], 48, 48, 1)

# init model

model__Affect = initModelTraining((48, 48, 1), num_labels)

# training model
model__Affect.fit(X_train_affect, train_y_affect,
                  batch_size=batch_size,
                  epochs=num_epochs,
                  verbose=1,
                  validation_data=(X_test_affect, test_y_affect),
                  shuffle=True)

# Saving the  model
fer_json = model__Affect.to_json()
with open("ferAffect.json", "w") as json_file:
    json_file.write(fer_json)
model__Affect.save_weights("ferAffect.h5")
