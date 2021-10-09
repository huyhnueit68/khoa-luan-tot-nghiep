# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
import glob
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

"""
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

"""


# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

def image_to_array(image, switch=False):
    if not switch:
        # load the image
        img = load_img(image)
        # print("Orignal:" ,type(img))
        # convert to numpy array
        img_array = img_to_array(img)
        return img_array
    else:
        return array_to_img(image)


def get_emo_array(i, maxim=0, j=1):
    image_list = []
    print('emotion ', i + 1, '-------------------------------------')
    total = len(glob.glob('../data/AffectNet/' + dirs[i] + '/*.jpg'))
    c, n = 1, 0
    for filename in glob.glob('../input/affectnetsample/val_class/' + dirs[i] + '/*.jpg'):
        if c >= j:
            # assuming gif
            print(n, 'images out of ', max(j - maxim + 1, maxim - j + 1))
            # printProgressBar(j,total,"")
            # im=image_to_array(filename)
            image_list.append(filename)
            n += 1
        c += 1
        if c > maxim and maxim != 0: break
        # printProgressBar(j,total,"")
    print(n, ' images uploaded successfully !!')
    return image_list


dicti = {0: 'neut', 1: 'happ', 2: 'sad', 3: 'surp', 4: 'fear', 5: 'dist', 6: 'ang', 7: 'contp'}


def f(x):
    return x.ravel()


def array_map(x):
    return list(map(f, np.array(x)))