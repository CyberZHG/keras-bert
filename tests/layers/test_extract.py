import unittest
import numpy as np
from keras_bert.backend import keras
from keras_bert.layers import Extract


class TestExtract(unittest.TestCase):

    def test_sample(self):
        input_layer = keras.layers.Input(
            shape=(3, 4),
            name='Input',
        )
        extract_layer = Extract(
            index=0,
            name='Extract'
        )(input_layer)
        model = keras.models.Model(
            inputs=input_layer,
            outputs=extract_layer,
        )
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics={},
        )
        model.summary()
        inputs = np.asarray([[
            [0.1, 0.2, 0.3, 0.4],
            [-0.1, 0.2, -0.3, 0.4],
            [0.1, -0.2, 0.3, -0.4],
        ]])
        predict = model.predict(inputs)
        expected = np.asarray([[0.1, 0.2, 0.3, 0.4]])
        self.assertTrue(np.allclose(expected, predict), predict)
