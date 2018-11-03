import unittest
import os
from keras_bert import load_trained_model_from_checkpoint


class TestLoader(unittest.TestCase):

    def test_load_trained(self):
        current_path = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_path, 'test_checkpoint', 'config.json')
        model_path = os.path.join(current_path, 'test_checkpoint', 'model.ckpt-20')
        model = load_trained_model_from_checkpoint(config_path, model_path, training=False)
        model.summary()

    def test_load_training(self):
        current_path = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_path, 'test_checkpoint', 'config.json')
        model_path = os.path.join(current_path, 'test_checkpoint', 'model.ckpt-20')
        model = load_trained_model_from_checkpoint(config_path, model_path, training=True)
        model.summary()
