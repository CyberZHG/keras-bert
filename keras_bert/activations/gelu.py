import math
import keras.backend as K


def gelu(x):
    """An approximation of gelu.

    See: https://arxiv.org/pdf/1606.08415.pdf
    """
    if K.backend() == 'tensorflow':
        import tensorflow as tf
        return 0.5 * x * (1.0 + tf.erf(x / tf.sqrt(2.0)))
    return 0.5 * x * (1.0 + K.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * K.pow(x, 3))))
