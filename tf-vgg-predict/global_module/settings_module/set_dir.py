import os

class Directory():
    def __init__(self):
        self.root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        self.sample_test_dir = self.root_path + '/data/sample_test'
        self.imagenet_class_dir = self.root_path + '/data/imagenet_classes.txt'
        self.weights_dir = self.root_path + '/data/weights.npz'