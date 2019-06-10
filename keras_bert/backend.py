import os

__all__ = [
    'keras', 'utils', 'activations', 'applications', 'backend', 'datasets', 'engine',
    'layers', 'preprocessing', 'wrappers', 'callbacks', 'constraints', 'initializers',
    'metrics', 'models', 'losses', 'optimizers', 'regularizers', 'TF_KERAS', 'EAGER_MODE'
]

TF_KERAS = False
EAGER_MODE = False

if 'TF_KERAS' in os.environ and os.environ['TF_KERAS'] != '0':
    import tensorflow as tf
    from tensorflow.python import keras
    TF_KERAS = True
    if 'TF_EAGER' in os.environ and os.environ['TF_EAGER'] != '0':
        if int(tf.version.VERSION.split('.')[0]) < 2:
            import tensorflow as tf
            tf.enable_eager_execution()
        EAGER_MODE = True
else:
    import keras

utils = keras.utils
activations = keras.activations
applications = keras.applications
backend = keras.backend
datasets = keras.datasets
engine = keras.engine
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
