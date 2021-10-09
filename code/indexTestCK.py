# import libraly necessary librari to run program include:
# 1. pip install opencv-python
# 2. pip install keras
import os

import cv2
from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np

def read_text_file(file_path):
    with open(file_path, 'r') as f:
        print(f.read())

# load model
model = model_from_json(open("ferTest.json", "r").read())

# load weights
model.load_weights('ferTest.h5')

# Folder Path
path = "./fear"
# Change the directory
os.chdir(path)
dataRead = []
# iterate through all file
for file in os.listdir():
    # Check whether file is in text format or not
    if file.endswith(".png"):
        file_path = f"{path}\{file}"
        cap = cv2.imread("."+file_path)
        # call read text file function
        dataRead.append(cap)

emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

lable_img, input_img = [], []
count = 0
i = 0
result_Img = 0
x, y, w, h = 48, 48, 48, 48

for image_value in dataRead:
    try:
        gray_img = cv2.cvtColor(image_value, cv2.COLOR_BGR2GRAY)
        # conver a PIL Image instance to a Numpy array.
        img_pixels = image.img_to_array(gray_img)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        # find max indexed array, returns the indices of the maximum values along an axis.
        max_index = np.argmax(predictions[0])

        if max_index == 2:
            result_Img += 1
        else:
            print(i)
        i += 1
    except:
        continue



print(result_Img)
print(i)
