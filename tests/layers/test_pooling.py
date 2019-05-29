from unittest import TestCase
import numpy as np
from keras_bert.backend import keras
from keras_bert.layers import MaskedGlobalMaxPool1D


class TestPooling(TestCase):

    def test_masked_global_max_pool_1d_predict(self):
        input_layer = keras.layers.Input(shape=(None,))
        embed_layer = keras.layers.Embedding(
            input_dim=5,
            output_dim=6,
            mask_zero=True,
            name='Embed'
        )(input_layer)
        pool_layer = MaskedGlobalMaxPool1D()(embed_layer)
        model = keras.models.Model(inputs=input_layer, outputs=pool_layer)
        model.compile(optimizer='adam', loss='mse')
        x = np.array([[1, 2, 0, 0], [2, 3, 4, 0]])
        y = model.predict(x)
        embed = model.get_layer('Embed').get_weights()[0]
        expected = np.max(embed[1:3], axis=0)
        self.assertTrue(np.allclose(expected, y[0]), (expected, y[0]))
        expected = np.max(embed[2:5], axis=0)
        self.assertTrue(np.allclose(expected, y[1]), (expected, y[1]))

    def test_masked_global_max_pool_1d_fit(self):
        input_layer = keras.layers.Input(shape=(None,))
        embed_layer = keras.layers.Embedding(
            input_dim=11,
            output_dim=13,
            mask_zero=False,
        )(input_layer)
        pool_layer = MaskedGlobalMaxPool1D()(embed_layer)
        dense_layer = keras.layers.Dense(units=2, activation='softmax')(pool_layer)
        model = keras.models.Model(inputs=input_layer, outputs=dense_layer)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        model.summary()
        x = np.random.randint(0, 11, (32, 7))
        y = np.random.randint(0, 2, (32,))
        model.fit(x, y)
