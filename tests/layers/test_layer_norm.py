import unittest
import keras
import numpy as np
from keras_bert.layers import LayerNormalization, get_multi_head_attention


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

    def test_fit(self):
        input_layer = keras.layers.Input(
            shape=(2, 3),
            name='Input',
        )
        att_layer = get_multi_head_attention(
            input_layer,
            head_num=3,
            dropout=1e-5,
            name='MH'
        )
        dense_layer = keras.layers.Dense(units=3, name='Dense-1')(att_layer)
        norm_layer = LayerNormalization(
            name='Layer-Normalization',
            trainable=False,
        )(dense_layer)
        dense_layer = keras.layers.Dense(units=3, name='Dense-2')(norm_layer)
        model = keras.models.Model(
            inputs=input_layer,
            outputs=dense_layer,
        )
        model.compile(
            optimizer=keras.optimizers.Adam(lr=1e-3),
            loss='mse',
            metrics=['mse'],
        )
        model.summary()

        def _generator(batch_size=32):
            while True:
                inputs = np.random.random((batch_size, 2, 3))
                outputs = np.asarray([[[0.0, -0.1, 0.2]] * 2] * batch_size)
                yield inputs, outputs

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
            expect = np.round(np.asarray([[[0.0, -0.1, 0.2]] * 2] * 3), decimals=1)
            actual = np.round(predicts, decimals=1)
            self.assertTrue(np.allclose(expect, actual), (expect, actual))
            break
