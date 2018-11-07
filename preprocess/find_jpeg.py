import numpy as np
import os

train_dir = '/home/yangyuhao/data/road/data/test_data/label_filter/data_50/good/'


def get_files(file_dir):
    name = []
    for file in os.listdir(file_dir):
        name.append(file_dir + file)
    print('There are %d image' % (len(name)))
    return name


a = get_files(train_dir)
b = len(a)
from PIL import Image


def is_jpg(filename):
    try:
        i = Image.open(filename)
        return i.format == 'JPG'
    except IOError:
        print('fuck you {}'.format(filename))
        return False


for i in range(b):
    is_jpg(a[i])

