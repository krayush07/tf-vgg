# caffe-to-tensorflow
This folder contains steps to migrate caffe model to tensorflow standards.

<br></br>

<b>How to extract weights and architecture from caffe model</b>

* Download [binary](http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel) of caffe model which contain weights learned during training.
* Next, download [layer configuration](https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/0067c9b32f60362c74f4c445a080beed06b07eb3/VGG_ILSVRC_16_layers_deploy.prototxt), more popularly known as prototxt file.
* Once trained weights and configuration file are downloaded, we can extract the network configuration and weights in numpy native format using the [script](https://github.com/ethereon/caffe-tensorflow/blob/master/convert.py).
* Sample command to run the script: `python convert.py [path-to-prototxt] [--caffemodel=path-to-caffe-binary] [--data-output=path-to-weight-file] [--code-output-path=path-to-network-configuration]`.
* One might experience some blockers while running the script. Please see next section that can be followed to successfully run the script.
* After running the above command, we obtain two files: **`data-output`** and **`code-output`** which provides the weights in numpy native format and network configuration (architecture) respectively.

<br></br>

<b> Steps to run end script to extract weights and architecture from caffe model</b>

* One might experience _unkown layer_ error while directly running the script. This error points to older version of caffe being used for training the model.
  * In case of above error, you need to upgrade the binary and prototxt files using [script<sup>1</sup>](https://github.com/BVLC/caffe/blob/master/tools/upgrade_net_proto_binary.cpp) and [script<sup>2</sup>](https://github.com/BVLC/caffe/blob/master/tools/upgrade_net_proto_text.cpp) provided in [caffe](https://github.com/BVLC/caffe) library.
  * To install Caffe (for Ubuntu 16.04 LTS), follow the steps exactly as mentioned [here](https://github.com/BVLC/caffe/wiki/Ubuntu-16.04-or-15.10-Installation-Guide). In case if you just need to install Caffe for upgradation of these two files, try to install CPU only version.
  * Once caffe is installed, you are ready to upgrade the files.

* Depending on installation environment, one might encounter another issue from protobuf which reads: _`google.protobuf.message.DecodeError: Error parsing message`_. There are some turnaround suggested for this error. Final two step solution that worked for me in this case:
  * Setting protobuf variable as `export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`
  * Renaming `caffepb.py` to `caffe_pb2.py` and resolving the import error in `resolver.py`, all files inside [/kaffe/caffe/](https://github.com/ethereon/caffe-tensorflow/tree/master/kaffe/caffe) directory.

* Finally run 
  * `[caffe-root-path]/build/tools/upgrade_net_proto_binary [path-to-old-caffemodel] [path-to-new-caffemodel]`
  * `[caffe-root-path]/build/tools/upgrade_net_proto_text [path-to-old-prototxt] [path-to-new-prototxt]`
