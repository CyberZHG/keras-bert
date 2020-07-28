from keras_bert.backend import keras


class Extract(keras.layers.Layer):
    """Extract from index.

    See: https://arxiv.org/pdf/1810.04805.pdf
    """

    def __init__(self, index, **kwargs):
        super(Extract, self).__init__(**kwargs)
        self.index = index
        self.supports_masking = True

    def get_config(self):
        config = {
            'index': self.index,
        }
        base_config = super(Extract, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, x, mask=None):
        return x[:, self.index]
