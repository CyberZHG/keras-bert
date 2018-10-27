import unittest
import keras
import numpy as np
from keras_bert.layers import get_inputs, Embeddings, Transformer, Masked


class TestMasked(unittest.TestCase):

    def test_sample(self):
        inputs = get_inputs(seq_len=512)
        embed_layer = Embeddings(input_dim=12, output_dim=768, position_dim=512)(inputs)
        masked_layer = Masked(name='Masked')([embed_layer, inputs[-1]])
        model = keras.models.Model(inputs=inputs, outputs=masked_layer)
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics={},
        )
        model.summary(line_length=120)
        model.predict([
            np.asarray([[1] + [0] * 511]),
            np.asarray([[0] * 512]),
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
        transformer_layer = Transformer(
            head_num=1,
            hidden_dim=12,
            dropout_rate=0.1,
            name='Transformer',
        )(embed_layer)
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
            metrics=['mse'],
        )
        model.summary(line_length=150)
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
