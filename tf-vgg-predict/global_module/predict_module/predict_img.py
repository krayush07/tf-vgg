import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
from global_module.settings_module import set_dir
import os
import matplotlib.pyplot as plt


def read_image(img_name, img_mode='RGB'):
    return imread(name=img_name, mode=img_mode)


def resize_image(img_arr, img_size=(256, 256)):
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


class ImagePredictVGG():
    def __init__(self):
        self.network_architecture()

    def transform_image(self, img):
        self.transformed_img = img

    def get_conv_layer(self, name, input):
        with tf.variable_scope(name) as scope:
            kernel_filter = self.get_layer_weight(name)
            bias = self.get_layer_bias(name)
            convoluted_output = tf.nn.conv2d(input=input, filter=kernel_filter, strides=[1, 1, 1, 1], padding='SAME', name=scope)
            feature = tf.nn.bias_add(convoluted_output, bias, name=scope)
            return tf.nn.relu(feature, name=scope)

    def get_fully_connected_layer(self, name, input):
        with tf.variable_scope(name) as scope:
            weights = self.get_layer_weight(name)
            bias = self.get_layer_bias(name)

            shape = input.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            value = tf.reshape(input, [-1, dim])
            return tf.nn.bias_add(tf.matmul(value, weights), bias)

    def get_max_pool_layer(self, name, input):
        return tf.nn.max_pool(input, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def get_layer_weight(self, name):
        return tf.constant(self.network_param[name][0], name='weights')

    def get_layer_bias(self, name):
        return tf.constant(self.network_param[name][1], name='biases')

    def network_architecture(self):
        conv1_1 = self.get_conv_layer('conv1_1', self.transformed_img)
        conv1_2 = self.get_conv_layer('conv1_2', conv1_1)
        pool1 = self.get_max_pool_layer('pool1', conv1_2)

        conv2_1 = self.get_conv_layer('conv2_1', pool1)
        conv2_2 = self.get_conv_layer('conv2_2', conv2_1)
        pool2 = self.get_max_pool_layer('pool2', conv2_2)

        conv3_1 = self.get_conv_layer('conv3_1', pool2)
        conv3_2 = self.get_conv_layer('conv3_2', conv3_1)
        conv3_3 = self.get_conv_layer('conv3_3', conv3_2)
        pool3 = self.get_max_pool_layer('pool3', conv3_3)

        conv4_1 = self.get_conv_layer('conv4_1', pool3)
        conv4_2 = self.get_conv_layer('conv4_2', conv4_1)
        conv4_3 = self.get_conv_layer('conv4_3', conv4_2)
        pool4 = self.get_max_pool_layer('pool4', conv4_3)

        conv5_1 = self.get_conv_layer('conv5_1', pool4)
        conv5_2 = self.get_conv_layer('conv5_2', conv5_1)
        conv5_3 = self.get_conv_layer('conv5_3', conv5_2)
        pool5 = self.get_max_pool_layer('pool5', conv5_3)

        fc6 = self.get_fully_connected_layer('fc6', pool5)
        fc7 = self.get_fully_connected_layer('fc7', tf.nn.relu(fc6))
        fc8 = self.get_fully_connected_layer('fc8', tf.nn.relu(fc7))

        prob = tf.nn.softmax(fc8, name='prob')

def main():
    set_dir_obj = set_dir.Directory()
    test(set_dir_obj)


if __name__ == '__main__':
    main()
