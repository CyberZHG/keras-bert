# coding=utf-8
from __future__ import unicode_literals
import unittest
import os
import codecs
from keras_bert.backend import keras
from keras_bert import get_model, POOL_NSP, POOL_MAX, POOL_AVE, extract_embeddings


class TestUtil(unittest.TestCase):

    def setUp(self):
        current_path = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(current_path, 'test_checkpoint')

    def test_extract_embeddings_default(self):
        embeddings = extract_embeddings(self.model_path, ['all work and no play', 'makes jack a dull boy~'])
        self.assertEqual(2, len(embeddings))
        self.assertEqual((7, 4), embeddings[0].shape)
        self.assertEqual((8, 4), embeddings[1].shape)

    def test_extract_embeddings_batch_size_1(self):
        embeddings = extract_embeddings(
            self.model_path,
            ['all work and no play', 'makes jack a dull boy~'],
            batch_size=1,
        )
        self.assertEqual(2, len(embeddings))
        self.assertEqual((7, 4), embeddings[0].shape)
        self.assertEqual((8, 4), embeddings[1].shape)

    def test_extract_embeddings_pair(self):
        embeddings = extract_embeddings(
            self.model_path,
            [
                ('all work and no play', 'makes jack a dull boy'),
                ('makes jack a dull boy', 'all work and no play'),
            ],
        )
        self.assertEqual(2, len(embeddings))
        self.assertEqual((13, 4), embeddings[0].shape)

    def test_extract_embeddings_single_pooling(self):
        embeddings = extract_embeddings(
            self.model_path,
            [
                ('all work and no play', 'makes jack a dull boy'),
                ('makes jack a dull boy', 'all work and no play'),
            ],
            poolings=POOL_NSP,
        )
        self.assertEqual(2, len(embeddings))
        self.assertEqual((4,), embeddings[0].shape)

    def test_extract_embeddings_multi_pooling(self):
        embeddings = extract_embeddings(
            self.model_path,
            [
                ('all work and no play', 'makes jack a dull boy'),
                ('makes jack a dull boy', 'all work and no play'),
            ],
            poolings=[POOL_NSP, POOL_MAX, POOL_AVE],
            output_layer_num=2,
        )
        self.assertEqual(2, len(embeddings))
        self.assertEqual((24,), embeddings[0].shape)

    def test_extract_embeddings_invalid_pooling(self):
        with self.assertRaises(ValueError):
            extract_embeddings(
                self.model_path,
                [
                    ('all work and no play', 'makes jack a dull boy'),
                    ('makes jack a dull boy', 'all work and no play'),
                ],
                poolings=['invalid'],
            )

    def test_extract_embeddings_variable_lengths(self):
        tokens = [
            '[PAD]', '[UNK]', '[CLS]', '[SEP]',
            'all', 'work', 'and', 'no', 'play',
            'makes', 'jack', 'a', 'dull', 'boy', '~',
        ]
        token_dict = {token: i for i, token in enumerate(tokens)}
        inputs, outputs = get_model(
            token_num=len(tokens),
            pos_num=20,
            seq_len=None,
            embed_dim=13,
            transformer_num=1,
            feed_forward_dim=17,
            head_num=1,
            training=False,
        )
        model = keras.models.Model(inputs, outputs)
        embeddings = extract_embeddings(
            model,
            [
                ('all work and no play', 'makes jack'),
                ('a dull boy', 'all work and no play and no play'),
            ],
            vocabs=token_dict,
            batch_size=2,
        )
        self.assertEqual(2, len(embeddings))
        self.assertEqual((10, 13), embeddings[0].shape)
        self.assertEqual((14, 13), embeddings[1].shape)

    def test_extract_embeddings_from_file(self):
        with codecs.open(os.path.join(self.model_path, 'vocab.txt'), 'r', 'utf8') as reader:
            texts = map(lambda x: x.strip(), reader)
            embeddings = extract_embeddings(self.model_path, texts)
        self.assertEqual(15, len(embeddings))
