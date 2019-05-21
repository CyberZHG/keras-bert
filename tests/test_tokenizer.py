# coding=utf-8
from __future__ import unicode_literals
from unittest import TestCase
from keras_bert import Tokenizer


class TestTokenizer(TestCase):

    def test_uncased(self):
        tokens = [
            '[PAD]', '[UNK]', '[CLS]', '[SEP]', 'want', '##want',
            '##ed', 'wa', 'un', 'runn', '##ing', ',',
            '\u535A', '\u63A8',
        ]
        token_dict = {token: i for i, token in enumerate(tokens)}
        tokenizer = Tokenizer(token_dict)
        text = u"UNwant\u00E9d, running  \nah\u535A\u63A8zzz\u00AD"
        tokens = tokenizer.tokenize(text)
        expected = [
            '[CLS]', 'un', '##want', '##ed', ',', 'runn', '##ing',
            'a', '##h', '\u535A', '\u63A8', 'z', '##z', '##z',
            '[SEP]',
        ]
        self.assertEqual(expected, tokens)
        indices, segments = tokenizer.encode(text)
        expected = [2, 8, 5, 6, 11, 9, 10, 1, 1, 12, 13, 1, 1, 1, 3]
        self.assertEqual(expected, indices)
        expected = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.assertEqual(expected, segments)

        decoded = tokenizer.decode(indices)
        expected = [
            'un', '##want', '##ed', ',', 'runn', '##ing',
            '[UNK]', '[UNK]', '\u535A', '\u63A8', '[UNK]', '[UNK]', '[UNK]',
        ]
        self.assertEqual(expected, decoded)

    def test_padding(self):
        tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]']
        token_dict = {token: i for i, token in enumerate(tokens)}
        tokenizer = Tokenizer(token_dict)
        text = '\u535A\u63A8'

        # single
        indices, segments = tokenizer.encode(first=text, max_len=100)
        expected = [2, 1, 1, 3] + [0] * 96
        self.assertEqual(expected, indices)
        expected = [0] * 100
        self.assertEqual(expected, segments)
        decoded = tokenizer.decode(indices)
        self.assertEqual(['[UNK]', '[UNK]'], decoded)
        indices, segments = tokenizer.encode(first=text, max_len=3)
        self.assertEqual([2, 1, 3], indices)
        self.assertEqual([0, 0, 0], segments)

        # paired
        indices, segments = tokenizer.encode(first=text, second=text, max_len=100)
        expected = [2, 1, 1, 3, 1, 1, 3] + [0] * 93
        self.assertEqual(expected, indices)
        expected = [0, 0, 0, 0, 1, 1, 1] + [0] * 93
        self.assertEqual(expected, segments)
        decoded = tokenizer.decode(indices)
        self.assertEqual((['[UNK]', '[UNK]'], ['[UNK]', '[UNK]']), decoded)
        indices, segments = tokenizer.encode(first=text, second=text, max_len=4)
        self.assertEqual([2, 1, 3, 3], indices)
        self.assertEqual([0, 0, 0, 1], segments)

    def test_empty(self):
        tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]']
        token_dict = {token: i for i, token in enumerate(tokens)}
        tokenizer = Tokenizer(token_dict)
        text = u''
        self.assertEqual(['[CLS]', '[SEP]'], tokenizer.tokenize(text))
        indices, segments = tokenizer.encode(text)
        self.assertEqual([2, 3], indices)
        self.assertEqual([0, 0], segments)
        decoded = tokenizer.decode(indices)
        self.assertEqual([], decoded)

    def test_cased(self):
        tokens = [
            '[UNK]', u'[CLS]', '[SEP]', 'want', '##want',
            '##\u00E9d', 'wa', 'UN', 'runn', '##ing', ',',
        ]
        token_dict = {token: i for i, token in enumerate(tokens)}
        tokenizer = Tokenizer(token_dict, cased=True)
        text = "UNwant\u00E9d, running"
        tokens = tokenizer.tokenize(text)
        expected = ['[CLS]', 'UN', '##want', '##\u00E9d', ',', 'runn', '##ing', '[SEP]']
        self.assertEqual(expected, tokens)
        indices, segments = tokenizer.encode(text)
        expected = [1, 7, 4, 5, 10, 8, 9, 2]
        self.assertEqual(expected, indices)
        expected = [0, 0, 0, 0, 0, 0, 0, 0]
        self.assertEqual(expected, segments)
