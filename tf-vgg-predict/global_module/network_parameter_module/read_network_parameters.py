import numpy as np


def read_network_params(filename):
    weights = np.load(filename)
    each_layer = weights.item()
    output_file = open('params.out', 'w')
    for idx, curr_layeri in enumerate(each_layer.keys()):
        param_str = ''
        curr_layer = each_layer[each_layer.keys()[idx]]
        param_str += each_layer.keys()[idx] + '\t'
        for each_key in curr_layer.keys():
            param_str += each_key + '\t'
        print param_str.strip()
        output_file.write(param_str.strip() + '\n')
    output_file.close()


read_network_params('/home/aykumar/aykumar_home/self/vgg/caffe-tensorflow-master/data.npz')
