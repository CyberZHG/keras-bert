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
        expected = np.asarray([[[0.03814463, 0.03196599, -0.15434355]]])
        self.assertTrue(np.allclose(expected, predict), predict)

    def test_fit(self):
        input_layer = keras.layers.Input(
            shape=(1, 3),
            name='Input',
        )
        feed_forward_layer = FeedForward(
            hidden_dim=4,
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

        def _generator(batch_size=32):
            while True:
                inputs = np.random.random((batch_size, 1, 3))
                outputs = inputs * 0.8 + 0.3
                yield inputs, outputs

        for _ in range(3):
            model.fit_generator(
                generator=_generator(),
                steps_per_epoch=1000,
                epochs=30,
                validation_data=_generator(),
                validation_steps=100,
                callbacks=[
                    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
                ],
            )
            for inputs, _ in _generator(batch_size=3):
                predicts = model.predict(inputs)
                expect = np.round(inputs * 0.8 + 0.3, decimals=1)
                actual = np.round(predicts, decimals=1)
                if np.allclose(expect, actual):
                    return
                break
        self.assertTrue(np.allclose(expect, actual), (expect, actual))
