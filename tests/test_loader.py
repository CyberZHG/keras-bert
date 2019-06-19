import unittest
import os
from keras_bert import load_trained_model_from_checkpoint, load_vocabulary


class TestLoader(unittest.TestCase):

    def test_load_trained(self):
        current_path = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_path, 'test_checkpoint', 'bert_config.json')
        model_path = os.path.join(current_path, 'test_checkpoint', 'bert_model.ckpt')
        model = load_trained_model_from_checkpoint(config_path, model_path, training=False)
        model.summary()

    def test_load_trained_shorter(self):
        current_path = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_path, 'test_checkpoint', 'bert_config.json')
        model_path = os.path.join(current_path, 'test_checkpoint', 'bert_model.ckpt')
        model = load_trained_model_from_checkpoint(config_path, model_path, training=False, seq_len=8)
        model.summary()

    def test_load_training(self):
        current_path = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_path, 'test_checkpoint', 'bert_config.json')
        model_path = os.path.join(current_path, 'test_checkpoint', 'bert_model.ckpt')
        model = load_trained_model_from_checkpoint(config_path, model_path, training=True)
        model.summary()

    def test_load_output_layer_num(self):
        current_path = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_path, 'test_checkpoint', 'bert_config.json')
        model_path = os.path.join(current_path, 'test_checkpoint', 'bert_model.ckpt')
        model = load_trained_model_from_checkpoint(config_path, model_path, training=False, output_layer_num=4)
        model.summary()

    def test_load_with_trainable_prefixes(self):
        current_path = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_path, 'test_checkpoint', 'bert_config.json')
        model_path = os.path.join(current_path, 'test_checkpoint', 'bert_model.ckpt')
        model = load_trained_model_from_checkpoint(
            config_path,
            model_path,
            training=False,
            trainable=['Encoder'],
        )
        model.summary()

    def test_load_vocabulary(self):
        current_path = os.path.dirname(os.path.abspath(__file__))
        vocab_path = os.path.join(current_path, 'test_checkpoint', 'vocab.txt')
        token_dict = load_vocabulary(vocab_path)
        self.assertEqual(15, len(token_dict))
