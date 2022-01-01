import unittest

import numpy as np
from tensorflow import keras
from tensorflow.keras import backend as K

from keras_transformer import gelu, get_encoders
from keras_bert.layers import get_inputs, get_embedding, Masked


class TestMasked(unittest.TestCase):

    def test_sample(self):
        inputs = get_inputs(seq_len=512)
        embed_layer, _ = get_embedding(inputs, token_num=12, embed_dim=768, pos_num=512)
        masked_layer = Masked(name='Masked')([embed_layer, inputs[-1]])
        model = keras.models.Model(inputs=inputs, outputs=masked_layer)
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics={},
        )
        model.summary()
        model.predict([
            np.asarray([[1] + [0] * 511]),
            np.asarray([[0] * 512]),
            np.asarray([[1] + [0] * 511]),
        ])
        self.assertEqual((None, 512, 768), model.layers[-1].output_shape)

    def test_mask_result(self):
        input_layer = keras.layers.Input(
            shape=(None,),
            name='Input',
        )
        embed_layer = keras.layers.Embedding(
            input_dim=12,
            output_dim=9,
            mask_zero=True,
            name='Embedding',
        )(input_layer)
        transformer_layer = get_encoders(
            encoder_num=1,
            input_layer=embed_layer,
            head_num=1,
            hidden_dim=12,
            attention_activation=None,
            feed_forward_activation=gelu,
            dropout_rate=0.1,
        )
        dense_layer = keras.layers.Dense(
            units=12,
            activation='softmax',
            name='Dense',
        )(transformer_layer)
        mask_layer = keras.layers.Input(
            shape=(None,),
            name='Mask',
        )
        masked_layer, mask_result = Masked(
            return_masked=True,
            name='Masked',
        )([dense_layer, mask_layer])
        model = keras.models.Model(
            inputs=[input_layer, mask_layer],
            outputs=[masked_layer, mask_result],
        )
        model.compile(
            optimizer='adam',
            loss='mse',
        )
        model.summary()
        predicts = model.predict([
            np.asarray([
                [1, 2, 3, 4, 5, 6, 7, 8, 0, 0],
                [1, 2, 3, 4, 0, 0, 0, 0, 0, 0],
            ]),
            np.asarray([
                [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
            ]),
        ])
        expect = np.asarray([
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        ])
        self.assertTrue(np.allclose(expect, predicts[1]))

    def test_mask_loss(self):
        def _loss(y_true, _):
            return K.sum(y_true, axis=-1)

        inputs = [keras.layers.Input((5,)), keras.layers.Input((5,))]
        embed = keras.layers.Embedding(input_dim=2, output_dim=3, mask_zero=True)(inputs[0])
        masked = Masked()([embed, inputs[1]])

        model = keras.models.Model(inputs, masked)
        model.compile(
            optimizer='sgd',
            loss=_loss,
        )

        token_input = np.array([
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
        ])
        mask_input = np.array([
            [0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
        ])
        outputs = np.arange(30, dtype=K.floatx()).reshape((2, 5, 3))
        actual = model.evaluate([token_input, mask_input], outputs)
        self.assertTrue(np.abs(actual - 6.0) < 1e-6 or np.abs(actual - 30.0) < 1e-6, actual)
