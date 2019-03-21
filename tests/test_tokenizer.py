# coding=utf-8
from unittest import TestCase
from keras_bert import Tokenizer


class TestTokenizer(TestCase):

    def test_case(self):
        tokens = [
            '[UNK]', '[CLS]', '[SEP]', 'want', '##want',
            '##ed', 'wa', 'un', 'runn', '##ing', ',',
            u'\u535A', u'\u63A8',
        ]
        token_dict = {token: i for i, token in enumerate(tokens)}
        tokenizer = Tokenizer(token_dict)
        text = u"UNwant\u00E9d, running  \nah\u535A\u63A8zzz\u00AD"
        tokens = tokenizer.tokenize(text)
        expected = [
            '[CLS]', 'un', '##want', '##ed', ',', 'runn', '##ing',
            'a', '##h', u'\u535A', u'\u63A8', 'z', '##z', '##z',
            '[SEP]',
        ]
        self.assertEqual(expected, tokens)
        indices, segments = tokenizer.encode(text)
        expected = [1, 7, 4, 5, 10, 8, 9, 0, 0, 11, 12, 0, 0, 0, 2]
        self.assertEqual(expected, indices)
        expected = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.assertEqual(expected, segments)

    def test_padding(self):
        tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]']
        token_dict = {token: i for i, token in enumerate(tokens)}
        tokenizer = Tokenizer(token_dict)
        text = u'\u535A\u63A8'
        indices, segments = tokenizer.encode(first=text, second=text, max_len=100)
        expected = [2, 1, 1, 3, 1, 1, 3] + [0] * 93
        self.assertEqual(expected, indices)
        expected = [0, 0, 0, 0, 1, 1, 1] + [0] * 93
        self.assertEqual(expected, segments)
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
