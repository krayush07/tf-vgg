import numpy as np


def get_network_params(weight_filename):
    weights = np.load(weight_filename)
    each_layer = weights.item()

    param_dict = {}

    for idx, curr_layeri in enumerate(each_layer.keys()):
        curr_layer_name = each_layer.keys()[idx]
        param_dict[curr_layer_name] = {}
        curr_layer = each_layer[each_layer.keys()[idx]]
        for each_key in curr_layer.keys():
            param_dict[curr_layer_name][each_key] = curr_layer[each_key]
    return param_dict

def read_network_params(filename):
    weights = np.load(filename)
    output_file = open('params.out', 'w')
    each_layer = weights.item()
    for idx, curr_layer in enumerate(each_layer.keys()):
        param_str = ''
        curr_layer = each_layer[each_layer.keys()[idx]]
        param_str += each_layer.keys()[idx] + '\t'
        for each_key in curr_layer.keys():
            param_str += each_key + '\t'
        print param_str.strip()
        output_file.write(param_str.strip() + '\n')
    output_file.close()


#read_network_params('/home/aykumar/aykumar_home/self/vgg/caffe-tensorflow-master/data.npz')
#get_network_params('/home/aykumar/aykumar_home/self/vgg/caffe-tensorflow-master/data.npz')