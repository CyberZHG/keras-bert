import keras
import keras.backend as K
from .wrapper import Wrapper


class Embeddings(Wrapper):
    """Get embedding layer.

    See: https://arxiv.org/pdf/1810.04805.pdf
    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 position_dim=512,
                 dropout_rate=0.1,
                 **kwargs):
        """Initialize the layer.

        :param input_dim: Number of tokens.
        :param output_dim: The dimension of all embedding layers.
        :param position_dim: Maximum position.
        :param dropout_rate: Dropout rate.
        """
        self.supports_masking = True
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.position_dim = position_dim
        self.dropout_rate = dropout_rate
        super(Embeddings, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'position_dim': self.position_dim,
            'dropout_rate': self.dropout_rate,
        }
        base_config = super(Embeddings, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape[0] + (self.output_dim,)

    def compute_mask(self, inputs, input_mask=None):
        return K.not_equal(inputs[0], 0)

    def build(self, input_shape):
        self.layers['Embedding-Token'] = keras.layers.Embedding(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            mask_zero=True,
            trainable=self.trainable,
            name='Embedding-Token',
        )
        self.layers['Embedding-Segment'] = keras.layers.Embedding(
            input_dim=2,
            output_dim=self.output_dim,
            trainable=self.trainable,
            name='Embedding-Segment',
        )
        self.layers['Embedding-Position'] = keras.layers.Embedding(
            input_dim=self.position_dim,
            output_dim=self.output_dim,
            trainable=self.trainable,
            name='Embedding-Position',
        )
        self.layers['Dropout-Token'] = keras.layers.Dropout(
            rate=self.dropout_rate,
            trainable=self.trainable,
            name='Dropout-Token',
        )
        self.layers['Dropout-Segment'] = keras.layers.Dropout(
            rate=self.dropout_rate,
            trainable=self.trainable,
            name='Dropout-Segment',
        )
        self.layers['Dropout-Position'] = keras.layers.Dropout(
            rate=self.dropout_rate,
            trainable=self.trainable,
            name='Dropout-Position',
        )
        self.layers['Embedding'] = keras.layers.Add(name='Embedding')
        self.layers['Embedding-Dropout'] = keras.layers.Dropout(
            rate=self.dropout_rate,
            trainable=self.trainable,
            name='Embedding-Dropout',
        )
        super(Embeddings, self).build(input_shape)

    def call(self, inputs, **kwargs):
        input_token, input_segment, input_position = inputs[:3]
        dropouts = [
            self.layers['Dropout-Token'](self.layers['Embedding-Token'](input_token)),
            self.layers['Dropout-Segment'](self.layers['Embedding-Segment'](input_segment)),
            self.layers['Dropout-Position'](self.layers['Embedding-Position'](input_position)),
        ]
        embed_layer = self.layers['Embedding'](dropouts)
        return self.layers['Embedding-Dropout'](embed_layer)
