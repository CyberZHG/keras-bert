from tensorflow.python.ops.math_ops import erf, sqrt

__all__ = ['gelu']


def gelu(x):
    return 0.5 * x * (1.0 + erf(x / sqrt(2.0)))
