import unittest

import numpy as np

from keras_bert.backend import keras
from keras_bert.backend import backend as K
from keras_bert.layers import TaskEmbedding


class TestTaskEmbedding(unittest.TestCase):

    def test_mask_zero(self):
        embed_input = keras.layers.Input(shape=(5, 4))
        task_input = keras.layers.Input(shape=(1,))
        task_embed = TaskEmbedding(input_dim=2, output_dim=4, mask_zero=True)([embed_input, task_input])
        func = K.function([embed_input, task_input], [task_embed])
        embed, task = np.random.random((2, 5, 4)), np.array([[0], [1]])
        output = func([embed, task])[0]
        self.assertTrue(np.allclose(embed[0], output[0]))
        self.assertFalse(np.allclose(embed[1], output[1]))
