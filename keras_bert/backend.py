import os
from distutils.util import strtobool

__all__ = [
    'keras', 'utils', 'activations', 'backend',
    'layers', 'callbacks', 'constraints', 'initializers',
    'metrics', 'models', 'losses', 'optimizers', 'regularizers', 'TF_KERAS',
]

TF_KERAS = strtobool(os.environ.get('TF_KERAS', '0'))

if TF_KERAS:
    import tensorflow as tf
    keras = tf.keras
else:
    import keras

utils = keras.utils
activations = keras.activations
backend = keras.backend
layers = keras.layers
callbacks = keras.callbacks
constraints = keras.constraints
initializers = keras.initializers
metrics = keras.metrics
models = keras.models
losses = keras.losses
optimizers = keras.optimizers
regularizers = keras.regularizers
