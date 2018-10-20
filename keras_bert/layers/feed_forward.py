import keras
import keras.backend as K


class FeedForward(keras.layers.Layer):
    """Position-wise feed-forward layer"""

    def __init__(self, hidden_dim, **kwargs):
        self.supports_masking = True
        self.hidden_dim = hidden_dim
        self.W1, self.b1 = None, None
        self.W2, self.b2 = None, None
        super(FeedForward, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'hidden_dim': self.hidden_dim,
        }
        base_config = super(FeedForward, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        return input_mask

    def build(self, input_shape):
        feature_dim = input_shape[-1]
        self.W1 = self.add_weight(shape=(feature_dim, self.hidden_dim),
                                  name='{}_W1'.format(self.name),
                                  initializer=keras.initializers.get('glorot_normal'))
        self.b1 = self.add_weight(shape=(self.hidden_dim,),
                                  name='{}_b1'.format(self.name),
                                  initializer=keras.initializers.get('zeros'))
        self.W2 = self.add_weight(shape=(self.hidden_dim, feature_dim),
                                  name='{}_W2'.format(self.name),
                                  initializer=keras.initializers.get('glorot_normal'))
        self.b2 = self.add_weight(shape=(feature_dim,),
                                  name='{}_b2'.format(self.name),
                                  initializer=keras.initializers.get('zeros'))
        super(FeedForward, self).build(input_shape)

    def call(self, x, mask=None):
        return K.dot(K.maximum(0.0, K.dot(x, self.W1) + self.b1), self.W2) + self.b2
