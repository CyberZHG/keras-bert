import keras
import keras.backend as K


class Masked(keras.layers.Layer):
    """Generate output mask based on the given mask.

    The inputs for the layer is the original input layer and the masked locations.

    See: https://arxiv.org/pdf/1810.04805.pdf
    """

    def __init__(self, **kwargs):
        self.supports_masking = True
        super(Masked, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def compute_mask(self, inputs, mask=None):
        token_mask = K.not_equal(inputs[1], 0)
        return K.all(K.stack([token_mask, mask[0]], axis=0), axis=0)

    def call(self, inputs, **kwargs):
        return inputs[0]
