import unittest
import keras
import numpy as np
from keras_bert.layers import LayerNormalization


class TestLayerNorm(unittest.TestCase):

    def test_sample(self):
        input_layer = keras.layers.Input(
            shape=(2, 3),
            name='Input',
        )
        norm_layer = LayerNormalization(
            name='Layer-Normalization',
        )(input_layer)
        model = keras.models.Model(
            inputs=input_layer,
            outputs=norm_layer,
        )
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mse'],
        )
        model.summary()
        inputs = np.array([[
            [0.2, 0.1, 0.3],
            [0.5, 0.1, 0.1],
        ]])
        predict = model.predict(inputs)
        expected = np.asarray([[
            [0.0, -1.22474487, 1.22474487],
            [1.41421356, -0.707106781, -0.707106781],
        ]])
        self.assertTrue(np.allclose(expected, predict))
