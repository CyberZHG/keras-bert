import unittest
import keras
from keras_bert.layers import get_inputs, get_embedding


class TestEmbedding(unittest.TestCase):

    def test_sample(self):
        inputs = get_inputs(seq_len=512)
        embed_layer = get_embedding(inputs, token_num=12, pos_num=512, embed_dim=768)
        model = keras.models.Model(inputs=inputs, outputs=embed_layer)
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mse'],
        )
        model.summary(line_length=120)
        self.assertEqual((None, 512, 768), model.layers[-1].output_shape)
