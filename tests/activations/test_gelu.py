import unittest
import keras.backend as K
from keras_bert.activations import gelu


class TestGelu(unittest.TestCase):

    def test_sample(self):
        results = gelu(K.constant([-30.0, -1.0, 0.0, 1.0, 30.0])).eval(session=K.get_session())
        self.assertEqual(0.0, results[0])
        self.assertGreater(0.0, results[1])
        self.assertLess(-1.0, results[1])
        self.assertEqual(0.0, results[2])
        self.assertGreater(1.0, results[3])
        self.assertLess(0.0, results[3])
        self.assertEqual(30.0, results[4])
