# import libraly necessary librari to run program include:
# 1. pip install opencv-python
# 2. pip install keras

import cv2
import pandas as pd
from PIL.Image import Image
from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import time

# load model
model = model_from_json(open("ferTest.json", "r").read())

# load weights
model.load_weights('ferTest.h5')

# read file csv
df = pd.read_csv('../data/Fer2013/fer2013PrivateTest.csv')

emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

lable_img, input_img = [], []
count = 0
for index, row in df.iterrows():
    try:
        if 'PrivateTest' in row['Usage']:
            image_string = row['pixels'].split(' ')  # pixels are separated by spaces.
            temp_img = np.asarray(image_string, dtype=np.uint8).reshape(48, 48)
            input_img.append(np.array(temp_img))
            lable_img.append(row['emotion'])
    except:
        print(f"error occured at index :{index} and row:{row}")

i = 0
result_Img = 0
x, y, w, h = 48, 48, 48, 48

timer = 0
count_face = 0
for image_value in input_img:

    # conver a PIL Image instance to a Numpy array.
    img_pixels = image.img_to_array(image_value)
    img_pixels = np.expand_dims(img_pixels, axis=0)
    img_pixels /= 255

    predictions = model.predict(img_pixels)

    start_time = time.time()
    predictions = model.predict(img_pixels)
    if count_face != 0:
        timer += time.time() - start_time
        print("--- Ảnh thứ " + str(count_face) + " %s seconds ---" % (time.time() - start_time))
    count_face += 1

    # find max indexed array, returns the indices of the maximum values along an axis.
    max_index = np.argmax(predictions[0])

    if lable_img[i] == max_index:
        result_Img += 1
    else:
        print(i)
    i += 1

count_face = count_face - 1
print("Trung bình" + str(timer/count_face))
print("Tổng thời gian: " + str(timer))
print("Du doan ket thuc")
print(result_Img)
print(i)
