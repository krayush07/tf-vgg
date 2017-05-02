import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
from global_module.settings_module import set_dir
import os
import matplotlib.pyplot as plt
from global_module.network_parameter_module import read_network_parameters

VGG_RGB_MEAN = [123.68, 116.779, 103.939]

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
    def __init__(self, weight_filename):
        self.network_param = read_network_parameters.get_network_params(weight_filename)
        self.rgb_mean = tf.constant(VGG_RGB_MEAN, dtype=tf.float32, name='rgb_mean')
        self.img_batch = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name='img_placeholder')
        self.transform_image()
        self.network_architecture()

    def transform_image(self):
        new_img_batch = tf.subtract(self.img_batch, self.rgb_mean)
        channels = tf.unstack(new_img_batch, axis=-1)
        self.transformed_img = tf.stack([channels[2], channels[1], channels[0]], axis = -1)

    def get_conv_layer(self, name, input):
        with tf.variable_scope(name) as scope:
            kernel_filter = self.get_layer_weight(name)
            bias = self.get_layer_bias(name)
            convoluted_output = tf.nn.conv2d(input=input, filter=kernel_filter, strides=[1, 1, 1, 1], padding='SAME', name=name+'_conv')
            feature = tf.nn.bias_add(convoluted_output, bias, name=name+'_bias')
            return tf.nn.relu(feature, name=name+'_relu')

    def get_fully_connected_layer(self, name, input):
        with tf.variable_scope(name) as scope:
            weights = self.get_layer_weight(name)
            bias = self.get_layer_bias(name)

            dim = tf.reduce_prod(tf.shape(input)[1:])
            value = tf.reshape(input, [-1, dim])
            return tf.nn.bias_add(tf.matmul(value, weights), bias)

    def get_max_pool_layer(self, name, input):
        return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def get_layer_weight(self, name):
        return tf.constant(self.network_param[name]['weights'], name='weights')

    def get_layer_bias(self, name):
        return tf.constant(self.network_param[name]['biases'], name='biases')

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

        self.prob = tf.nn.softmax(fc8, name='prob')

def get_imagenet_class():
    file = open(set_dir.Directory().imagenet_class_dir, 'r')
    class_arr = file.readlines()
    return class_arr

def main():
    set_dir_obj = set_dir.Directory()
    class_arr = get_imagenet_class()

    with tf.Session() as sess:
        img_vgg_obj = ImagePredictVGG(set_dir_obj.weights_dir)

        test_folder = set_dir_obj.sample_test_dir
        gold_label, images = [], []

        for filename in os.listdir(test_folder):
            img_arr = resize_image(read_image(test_folder + '/' + filename), img_size=(224, 224))
            images.append(img_arr)
            gold_label.append(filename)

        prob = sess.run(img_vgg_obj.prob, feed_dict={img_vgg_obj.img_batch: np.asarray(images)})
        for idx, each_prob in enumerate(prob):
            max = np.argmax(each_prob)
            plt.text(0.5, -4.5, 'Predicted: ' + class_arr[max].strip() + ', Prob.: ' + str(each_prob[max]), fontsize=12)
            plot_image(images[idx])
            print gold_label[idx], max, class_arr[max].strip(), each_prob[max]

if __name__ == '__main__':
    main()
