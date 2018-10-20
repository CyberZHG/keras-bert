import unittest
import os
import tempfile
import random
import keras
from keras_bert import get_model, get_custom_objects


class TestBERT(unittest.TestCase):

    def test_sample(self):
        model = get_model(
            token_num=200,
            head_num=3,
            transformer_num=2,
        )
        model_path = os.path.join(tempfile.gettempdir(), 'keras_bert_%f.h5' % random.random())
        model.save(model_path)
        model = keras.models.load_model(
            model_path,
            custom_objects=get_custom_objects(),
        )
        model.summary(line_length=200)
