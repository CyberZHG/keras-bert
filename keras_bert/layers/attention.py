import keras
import keras.backend as K


class Attention(keras.layers.Layer):
    """Attention layer.

    See: https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, **kwargs):
        self.supports_masking = True
        super(Attention, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape[-1]

    def compute_mask(self, inputs, mask=None):
        if isinstance(mask, list):
            return mask[-1]
        return mask

    def call(self, inputs, mask=None, **kwargs):
        query, key, value = inputs
        feature_dim = K.shape(query)[-1]
        e = K.batch_dot(query, key, axes=2) / K.sqrt(K.cast(feature_dim, dtype=K.floatx()))
        if isinstance(mask, list) and mask[-1] is not None:
            e -= (1.0 - K.cast(K.expand_dims(mask[-1]), K.floatx())) * 1e9
        a = keras.activations.softmax(e)
        v = K.batch_dot(a, value)
        return v
