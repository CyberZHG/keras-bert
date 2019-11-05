import unittest
from keras_bert.backend import keras
from keras_bert import compile_model, set_custom_objects
from keras_bert.albert import get_model


class TestALBERT(unittest.TestCase):

    def test_save_load_json_training(self):
        model = get_model(
            vocab_size=200,
            hidden_dropout_prob=0.1,
            training=True,
        )
        compile_model(model)
        data = model.to_json()
        set_custom_objects()
        model = keras.models.model_from_json(data)
        model.summary()

    def test_save_load_json_not_training(self):
        model = get_model(
            vocab_size=200,
            training=False,
            trainable=False,
            output_layers=[-1, -2, -3, -4],
        )
        compile_model(model)
        data = model.to_json()
        set_custom_objects()
        model = keras.models.model_from_json(data)
        model.summary()

        model = get_model(
            vocab_size=200,
            training=False,
            trainable=False,
            output_layers=-12,
        )
        compile_model(model)
        data = model.to_json()
        set_custom_objects()
        model = keras.models.model_from_json(data)
        model.summary()
