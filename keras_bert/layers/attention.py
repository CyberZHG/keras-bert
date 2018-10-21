import keras
import keras.backend as K


class Attention(keras.layers.Layer):
    """Self-attention layer.

    See: https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, **kwargs):
        self.supports_masking = True
        super(Attention, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, inputs, mask=None, **kwargs):
        feature_dim = K.shape(inputs)[-1]
        e = K.batch_dot(inputs, inputs, axes=2) / K.sqrt(K.cast(feature_dim, dtype=K.floatx()))
        if mask is not None:
            e -= (1.0 - K.cast(K.expand_dims(mask), K.floatx())) * 1e9
        a = keras.activations.softmax(e)
        v = K.batch_dot(a, inputs)
        return v
