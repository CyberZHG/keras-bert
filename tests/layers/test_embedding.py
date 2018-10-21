import unittest
import keras
from keras_bert.layers import get_inputs, Embeddings


class TestEmbedding(unittest.TestCase):

    def test_sample(self):
        inputs = get_inputs(seq_len=512)
        embed_layer = Embeddings(input_dim=12, output_dim=768, position_dim=512)(inputs)
        model = keras.models.Model(inputs=inputs, outputs=embed_layer)
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mse'],
        )
        model.summary(line_length=120)
        self.assertEqual((None, 512, 768), model.layers[-1].output_shape)
