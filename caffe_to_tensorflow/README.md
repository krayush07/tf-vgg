# caffe-to-tensorflow
This folder contains steps to migrate caffe model to tensorflow standards.

<b>How to extract weights and architecture from caffe model</b>

* Download [binary](http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel) of caffe model which contain weights learned during training.
* Next, download [layer configuration](https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/0067c9b32f60362c74f4c445a080beed06b07eb3/VGG_ILSVRC_16_layers_deploy.prototxt), more popularly known as prototxt file.
* Once trained weights and configuration file are downloaded, we can extract the network configuration and weights in numpy native format using the [script](https://github.com/ethereon/caffe-tensorflow/blob/master/convert.py).
* Sample command to run the script: *python convert.py [path-to-prototxt] [--caffemodel=path-to-caffe-binary] [--data-output=path-to-weight-file] [--code-output-path=path-to-network-configuration]*.
* One might experience some blockers while running the script. Please see next section that can be followed to successfully run the script.
* After running the above command, we obtain two files: **data-output** and **code-output** which provides the weights in numpy native format and network configuration (architecture) respectively.


<b> Steps to run end script to extract weights and architecture from caffe model</b>

* One might experience _unkown layer_ error while directly running the script. This error points to older version of caffe being used for training the model.
  * In case of above error, you need to upgrade the binary and prototxt files using script provided in caffe library.
