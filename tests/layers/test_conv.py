from unittest import TestCase
import numpy as np
from keras_bert.backend import keras
from keras_bert.layers import MaskedConv1D, MaskedGlobalMaxPool1D


class TestConv(TestCase):

    def test_masked_conv_1d_fit(self):
        input_layer = keras.layers.Input(shape=(7,))
        embed_layer = keras.layers.Embedding(
            input_dim=11,
            output_dim=13,
            mask_zero=True,
        )(input_layer)
        conv_layer = MaskedConv1D(filters=7, kernel_size=3)(embed_layer)
        pool_layer = MaskedGlobalMaxPool1D()(conv_layer)
        dense_layer = keras.layers.Dense(units=2, activation='softmax')(pool_layer)
        model = keras.models.Model(inputs=input_layer, outputs=dense_layer)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        model.summary()
        x = np.array(np.random.randint(0, 11, (32, 7)).tolist() * 100)
        y = np.array(np.random.randint(0, 2, (32,)).tolist() * 100)
        model.fit(x, y, epochs=10)
        y_hat = model.predict(x).argmax(axis=-1)
        self.assertEqual(y.tolist(), y_hat.tolist())
