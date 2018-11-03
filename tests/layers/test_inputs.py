import unittest
from keras_bert.layers import get_inputs


class TestInputs(unittest.TestCase):

    def test_name(self):
        inputs = get_inputs(seq_len=512)
        self.assertEqual(3, len(inputs))
        self.assertTrue('Segment' in inputs[1].name)
