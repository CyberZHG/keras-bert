import keras
import keras.backend as K


class LayerNormalization(keras.layers.Layer):
    """Layer normalization.

    See: https://arxiv.org/pdf/1607.06450.pdf
    """

    def __init__(self, **kwargs):
        self.supports_masking = True
        self.gamma, self.beta = None, None
        super(LayerNormalization, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        return input_mask

    def build(self, input_shape):
        self.gamma = self.add_weight(shape=input_shape[1:],
                                     name='{}_gamma'.format(self.name),
                                     initializer=keras.initializers.get('ones'))
        self.beta = self.add_weight(shape=input_shape[1:],
                                    name='{}_beta'.format(self.name),
                                    initializer=keras.initializers.get('zeros'))
        super(LayerNormalization, self).build(input_shape)

    def call(self, x, mask=None):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + K.epsilon()) + self.beta
