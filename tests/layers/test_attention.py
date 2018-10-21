import unittest
import keras
import numpy as np
from keras_bert.layers import Attention


class TestAttention(unittest.TestCase):

    def test_sample(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Embedding(
            input_dim=4,
            output_dim=5,
            mask_zero=True,
            weights=[
                np.asarray([
                    [0.1, 0.2, 0.3, 0.4, 0.5],
                    [0.2, 0.3, 0.4, 0.6, 0.5],
                    [0.4, 0.7, 0.2, 0.6, 0.9],
                    [0.3, 0.5, 0.8, 0.9, 0.1],
                ]),
            ],
            name='Embedding',
        ))
        model.add(Attention(name='Attention'))
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mse'],
        )
        model.summary()
        inputs = np.array([[1, 2, 3, 1, 0]])
        predict = model.predict(inputs)[0]
        self.assertTrue(np.allclose(predict[0], predict[3]))
        self.assertTrue(np.allclose(
            np.asarray([0.24973875, 0.41574854, 0.44609764, 0.644485, 0.47796533]),
            predict[2],
        ), predict[2])
