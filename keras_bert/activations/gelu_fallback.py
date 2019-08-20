import math
from keras_bert.backend import backend as K

__all__ = ['gelu']


def gelu(x):
    return 0.5 * x * (1.0 + K.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))
