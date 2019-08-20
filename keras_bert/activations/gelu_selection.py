from keras_bert.backend import TF_KERAS

__all__ = ['gelu']

if TF_KERAS:
    from .gelu_tensorflow import gelu
else:
    from .gelu_fallback import gelu
