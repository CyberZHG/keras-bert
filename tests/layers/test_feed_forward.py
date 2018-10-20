import unittest
import keras
import numpy as np
from keras_bert.layers import FeedForward


class TestAttention(unittest.TestCase):

    def test_sample(self):
        input_layer = keras.layers.Input(
            shape=(1, 3),
            name='Input',
        )
        feed_forward_layer = FeedForward(
            hidden_dim=4,
            weights=[
                np.asarray([
                    [0.1, 0.2, 0.3, 0.4],
                    [-0.1, 0.2, -0.3, 0.4],
                    [0.1, -0.2, 0.3, -0.4],
                ]),
                np.asarray([
                    0.0, -0.1, 0.2, -0.3,
                ]),
                np.asarray([
                    [0.1, 0.2, 0.3],
                    [-0.1, 0.2, -0.3],
                    [0.1, -0.2, 0.3],
                    [-0.1, 0.2, 0.3],
                ]),
                np.asarray([
                    0.0, 0.1, -0.2,
                ]),
            ],
            name='FeedForward',
        )(input_layer)
        model = keras.models.Model(
            inputs=input_layer,
            outputs=feed_forward_layer,
        )
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mse'],
        )
        model.summary()
        inputs = np.array([[[0.2, 0.1, 0.3]]])
        predict = model.predict(inputs)
        expected = np.asarray([[[0.036, 0.044, -0.092]]])
        self.assertTrue(np.allclose(expected, predict))
