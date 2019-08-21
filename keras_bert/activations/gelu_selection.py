from keras_bert.backend import backend as K

__all__ = ['gelu']

if K.backend() == 'tensorflow':
    from .gelu_tensorflow import gelu
else:
    from .gelu_fallback import gelu
