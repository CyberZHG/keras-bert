import keras
from keras_multi_head import MultiHeadAttention
from .layer_norm import LayerNormalization
from .feed_forward import FeedForward
from .wrapper import Wrapper
from ..activations import gelu


class Transformer(Wrapper):
    """Generate a set of transformer layers.

    See: https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self,
                 head_num,
                 hidden_dim,
                 dropout_rate=0.1,
                 **kwargs):
        """Initialize the layer.

        :param head_num: Number of heads.
        :param hidden_dim: Hidden dimension for feed forward layer.
        :param dropout_rate: Dropout rate.
        """
        self.supports_masking = True
        self.head_num = head_num
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        super(Transformer, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'head_num': self.head_num,
            'hidden_dim': self.hidden_dim,
            'dropout_rate': self.dropout_rate,
        }
        base_config = super(Transformer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        return input_mask

    def build(self, input_shape):
        layer = MultiHeadAttention(
            head_num=self.head_num,
            trainable=self.trainable,
            kernel_activation=gelu,
            name='%s-MultiHead' % self.name,
        )
        self.layers[layer.name] = layer
        layer = LayerNormalization(
            trainable=self.trainable,
            name='%s-MultiHead-Norm' % self.name,
        )
        self.layers[layer.name] = layer
        layer = keras.layers.Dropout(
            rate=self.dropout_rate,
            trainable=self.trainable,
            name='%s-MultiHead-Dropout' % self.name,
        )
        self.layers[layer.name] = layer
        layer = keras.layers.Add(
            trainable=self.trainable,
            name='%s-MultiHead-Add' % self.name,
        )
        self.layers[layer.name] = layer
        layer = FeedForward(
            hidden_dim=self.hidden_dim,
            trainable=self.trainable,
            name='%s-FeedForward' % self.name,
        )
        self.layers[layer.name] = layer
        layer = LayerNormalization(
            trainable=self.trainable,
            name='%s-FeedForward-Norm' % self.name,
        )
        self.layers[layer.name] = layer
        layer = keras.layers.Dropout(
            rate=self.dropout_rate,
            trainable=self.trainable,
            name='%s-FeedForward-Dropout' % self.name,
        )
        self.layers[layer.name] = layer
        layer = keras.layers.Add(
            trainable=self.trainable,
            name='%s-FeedForward-Add' % self.name,
        )
        self.layers[layer.name] = layer
        super(Transformer, self).build(input_shape)

    def call(self, inputs, mask=None):
        multi_head_layer = self.layers['%s-MultiHead' % self.name](inputs)
        multi_head_norm = self.layers['%s-MultiHead-Norm' % self.name](multi_head_layer)
        multi_head_dropout_layer = self.layers['%s-MultiHead-Dropout' % self.name](multi_head_norm)
        multi_head_residual_layer = self.layers['%s-MultiHead-Add' % self.name]([inputs, multi_head_dropout_layer])
        feed_forward_layer = self.layers['%s-FeedForward' % self.name](multi_head_residual_layer)
        feed_forward_norm = self.layers['%s-FeedForward-Norm' % self.name](feed_forward_layer)
        feed_forward_dropout_layer = self.layers['%s-FeedForward-Dropout' % self.name](feed_forward_norm)
        feed_forward_residual_layer = self.layers['%s-FeedForward-Add' % self.name](
            [multi_head_residual_layer, feed_forward_dropout_layer]
        )
        return feed_forward_residual_layer
