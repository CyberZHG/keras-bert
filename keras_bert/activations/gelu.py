import math
import keras.backend as K


def gelu(x):
    """An approximation of gelu."""
    return 0.5 * x * (1.0 + K.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * K.pow(x, 3))))
