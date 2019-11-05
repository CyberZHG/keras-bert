import os
from distutils.util import strtobool

__all__ = [
    'keras', 'utils', 'activations', 'applications', 'backend', 'datasets',
    'layers', 'preprocessing', 'wrappers', 'callbacks', 'constraints', 'initializers',
    'metrics', 'models', 'losses', 'optimizers', 'regularizers', 'TF_KERAS',
]

TF_KERAS = strtobool(os.environ.get('TF_KERAS', '0'))

if TF_KERAS:
    from tensorflow import keras
else:
    import keras

utils = keras.utils
activations = keras.activations
applications = keras.applications
backend = keras.backend
datasets = keras.datasets
layers = keras.layers
preprocessing = keras.preprocessing
wrappers = keras.wrappers
callbacks = keras.callbacks
constraints = keras.constraints
initializers = keras.initializers
metrics = keras.metrics
models = keras.models
losses = keras.losses
optimizers = keras.optimizers
regularizers = keras.regularizers
