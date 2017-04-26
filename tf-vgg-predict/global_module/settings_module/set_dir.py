import os
# from global_module.settings_module import set_config


class Directory():
    def __init__(self):
        self.root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        self.sample_test_dir = self.root_path + '/data/sample_test'