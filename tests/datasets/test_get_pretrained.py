import os
from unittest import TestCase
from keras_bert.datasets import get_pretrained, PretrainedList


class TestGetPretrained(TestCase):

    def test_get_pretrained(self):
        path = get_pretrained(PretrainedList.__test__)
        self.assertTrue(os.path.exists(os.path.join(path, 'README.md')))
