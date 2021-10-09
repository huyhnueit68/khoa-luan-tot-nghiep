# import library necessary

import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.utils import np_utils
from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

def initModelTraining(input__shape):
    # call model Sequential
    model_fer = Sequential()

    # Block-1
    # create a Sequential model incrementally
    model_fer.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input__shape))
    model_fer.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model_fer.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model_fer.add(Dropout(0.5))

    # Block-2
    # 2nd convolution layer, 2D convolution layer (spatial convolution over images)
    model_fer.add(Conv2D(64, (3, 3), activation='relu'))
    model_fer.add(Conv2D(64, (3, 3), activation='relu'))
    model_fer.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model_fer.add(Dropout(0.5))

    # Block-3
    # 3rd convolution layer
    model_fer.add(Conv2D(128, (3, 3), activation='relu'))
    model_fer.add(Conv2D(128, (3, 3), activation='relu'))
    model_fer.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model_fer.add(Flatten())  # Return a copy of the array collapsed into one dimension.

    # Block-4
    # fully connected neural networks
    model_fer.add(Dense(1024, activation='relu'))  # Activate linear unit Rectified
    model_fer.add(Dropout(0.2))
    model_fer.add(Dense(1024, activation='relu'))
    model_fer.add(Dropout(0.2))
    model_fer.add(Dense(num_labels, activation='softmax'))  # evaluate the categorical probabilities of the input data by softmax

    model_fer.summary()

    # Compliling the model
    model_fer.compile(loss=categorical_crossentropy,  # Computes the categorical crossentropy loss
                       optimizer=Adam(),  # Optimizer that implements the Adam algorithm
                       metrics=['accuracy'])  # judge the performance of your model

    return model_fer

# init size image and number of filters
num_features = 64
num_labels = 7
batch_size = 64
epochs = 50
width, height = 48, 48

# TRAINING DATASET OF FER2013
# connect and read file csv
df = pd.read_csv('../data/Fer2013/fer2013.csv')

# init 4 array for training
X_train, train_y, X_test, test_y = [], [], [], []

# load all info in df and append to array
for index, row in df.iterrows():
    # Split a string
    val = row['pixels'].split(" ")
    try:
        # add element to the end of array with data format as float
        if 'Training' in row['Usage']:
            X_train.append(np.array(val, 'float32'))
            train_y.append(row['emotion'])
        elif 'PublicTest' in row['Usage']:
            X_test.append(np.array(val, 'float32'))
            test_y.append(row['emotion'])
    except:
        print(f"error occured at index :{index} and row:{row}")

X_train = np.array(X_train, 'float32')
train_y = np.array(train_y, 'float32')
X_test = np.array(X_test, 'float32')
test_y = np.array(test_y, 'float32')

# convert array to vertor, using num_classes set total class
train_y = np_utils.to_categorical(train_y, num_classes=num_labels)
test_y = np_utils.to_categorical(test_y, num_classes=num_labels)

# Compute the arithmetic mean along the specified axis and Compute the standard deviation along the specified axis.
X_train -= np.mean(X_train, axis=0)  # normalize data between 0 and 1
X_train /= np.std(X_train, axis=0)

X_test -= np.mean(X_test, axis=0)
X_test /= np.std(X_test, axis=0)

# matrix transpose
X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)
X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)

# call model Sequential
model__fer = initModelTraining(X_train.shape[1:])

ResultModel = model__fer.fit(X_train, train_y,
               batch_size=batch_size,
               epochs=epochs,
               verbose=1,
               validation_data=(X_test, test_y),
               shuffle=True)

# Saving the  model
fer_json = model__fer.to_json()
with open("ferCombile.json", "w") as json_file:
    json_file.write(fer_json)
model__fer.save_weights("ferCombile.h5")
