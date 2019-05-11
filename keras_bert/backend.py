import os

__all__ = [
    'keras', 'utils', 'activations', 'applications', 'backend', 'datasets', 'engine',
    'layers', 'preprocessing', 'wrappers', 'callbacks', 'constraints', 'initializers',
    'metrics', 'models', 'losses', 'optimizers', 'regularizers',
]

if 'TF_KERAS' in os.environ and os.environ['TF_KERAS'] != '0':
    from tensorflow.python import keras
    from tensorflow.python.keras import utils
    from tensorflow.python.keras import activations
    from tensorflow.python.keras import applications
    from tensorflow.python.keras import backend
    from tensorflow.python.keras import datasets
    from tensorflow.python.keras import engine
    from tensorflow.python.keras import layers
    from tensorflow.python.keras import preprocessing
    from tensorflow.python.keras import wrappers
    from tensorflow.python.keras import callbacks
    from tensorflow.python.keras import constraints
    from tensorflow.python.keras import initializers
    from tensorflow.python.keras import metrics
    from tensorflow.python.keras import models
    from tensorflow.python.keras import losses
    from tensorflow.python.keras import optimizers
    from tensorflow.python.keras import regularizers
else:
    import keras
    from keras import utils
    from keras import activations
    from keras import applications
    from keras import backend
    from keras import datasets
    from keras import engine
    from keras import layers
    from keras import preprocessing
    from keras import wrappers
    from keras import callbacks
    from keras import constraints
    from keras import initializers
    from keras import metrics
    from keras import models
    from keras import losses
    from keras import optimizers
    from keras import regularizers
