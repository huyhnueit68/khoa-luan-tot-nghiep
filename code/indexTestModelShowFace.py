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

# load model
model = model_from_json(open("ferTest.json", "r").read())

# load weights
model.load_weights('ferTest.h5')

# read file csv
df = pd.read_csv('../data/Fer2013/fer2013PrivateTest.csv')

emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

lable_img, input_img = [], []
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
x, y, w, h = 10, 10, 28, 28

# define emotions
emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

for image_value in input_img:
    cv2.rectangle(image_value, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)
    cv2.imshow("abc", image_value)

    # cropping region of interest i.e. face area from  image
    roi_gray = image_value
    roi_gray = cv2.resize(roi_gray, (48, 48))
    cv2.imshow("Facial analysis ", roi_gray)

    # conver a PIL Image instance to a Numpy array.
    img_pixels = image.img_to_array(roi_gray)
    img_pixels = np.expand_dims(img_pixels, axis=0)
    img_pixels /= 255

    predictions = model.predict(img_pixels)

    # find max indexed array, returns the indices of the maximum values along an axis.
    max_index = np.argmax(predictions[0])
    predicted_emotion = emotions[max_index]

    # set emotional state
    cv2.putText(image_value, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    resized_img = cv2.resize(image_value, (1000, 700))
    cv2.imshow("Facial emotion analysis ", image_value)

for image_value in input_img:
    cv2.rectangle(image_value, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)

    # conver a PIL Image instance to a Numpy array.
    img_pixels = image.img_to_array(image_value)
    img_pixels = np.expand_dims(img_pixels, axis=0)
    img_pixels /= 255

    predictions = model.predict(img_pixels)

    # find max indexed array, returns the indices of the maximum values along an axis.
    max_index = np.argmax(predictions[0])
    predicted_emotion = emotions[max_index]

    # set emotional state
    cv2.putText(image_value, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow(str(i), image_value)

    if lable_img[i] == max_index:
        result_Img += 1
    else:
        print(i)
    i += 1

print(result_Img)
