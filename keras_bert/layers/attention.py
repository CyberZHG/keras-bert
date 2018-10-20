import keras
import keras.backend as K


class Attention(keras.layers.Layer):

    def __init__(self, **kwargs):
        self.supports_masking = True
        super(Attention, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, inputs, mask=None, **kwargs):
        input_shape = K.shape(inputs)
        input_len, feature_dim = input_shape[1], input_shape[2]
        e = K.batch_dot(inputs, K.permute_dimensions(inputs, (0, 2, 1))) / K.sqrt(K.cast(feature_dim, dtype=K.floatx()))
        e = K.exp(e)
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            mask = K.expand_dims(mask)
            e = K.permute_dimensions(K.permute_dimensions(e * mask, (0, 2, 1)) * mask, (0, 2, 1))

        s = K.sum(e, axis=-1)
        s = K.tile(K.expand_dims(s, axis=-1), K.stack([1, 1, input_len]))
        a = e / (s + K.epsilon())

        inputs = K.permute_dimensions(inputs, (0, 2, 1))
        v = K.permute_dimensions(K.batch_dot(inputs, K.permute_dimensions(a, (0, 2, 1))), (0, 2, 1))
        return v
