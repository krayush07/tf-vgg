import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
from global_module.settings_module import set_dir
import os
import matplotlib.pyplot as plt

def read_image(img_name, img_mode='RGB'):
    return imread(name=img_name, mode=img_mode)

def resize_image(img_arr, img_size=(256,256)):
    return imresize(img_arr, img_size)

def transform_image_by_mean(img_arr, mean_arr):
    return img_arr - mean_arr

def plot_image(img_arr):
    plt.imshow(img_arr)
    plt.show()

def test(set_dir_obj):
    test_folder = set_dir_obj.sample_test_dir
    images = []
    for filename in os.listdir(test_folder):
        img_arr = resize_image(read_image(test_folder + '/' + filename), img_size=(224, 224))
        images.append(img_arr)
        plot_image(img_arr)

class imgPredictVGG():
    def __init__(self):
        self.network_architecture()

    def network_architecture(self):
        self.load_parameters = []

def main():
    set_dir_obj = set_dir.Directory()
    test(set_dir_obj)


if __name__ == '__main__':
    main()