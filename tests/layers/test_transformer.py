import unittest
import keras
from keras_bert.layers import get_transformer


class TestMultiHead(unittest.TestCase):

    def test_sample(self):
        input_layer = keras.layers.Input(
            shape=(512,),
            name='Input',
        )
        embed_layer = keras.layers.Embedding(
            input_dim=12,
            output_dim=768,
            mask_zero=True,
            name='Embedding',
        )(input_layer)
        output_layer = get_transformer(
            inputs=embed_layer,
            head_num=12,
            hidden_dim=768 * 4,
            name='Transformer',
        )
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mse'],
        )
        model.summary(line_length=120)
        self.assertEqual((None, 512, 768), model.layers[-1].output_shape)
