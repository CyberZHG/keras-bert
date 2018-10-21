import keras
from .attention import Attention
from ..activations.gelu import gelu
from .wrapper import Wrapper


class MultiHeadAttention(Wrapper):
    """Generate multi-head attention layers.

    See: https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self,
                 head_num,
                 dropout_rate=0.1,
                 **kwargs):
        """Initialize the layer.

        :param head_num: Number of heads.
        :param dropout_rate: Dropout rate.
        :param feature_dim: The dimension of input feature.
        """
        self.supports_masking = True
        self.head_num = head_num
        self.dropout_rate = dropout_rate
        super(MultiHeadAttention, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'head_num': self.head_num,
            'dropout_rate': self.dropout_rate,
        }
        base_config = super(MultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        return input_mask

    def build(self, input_shape):
        feature_dim = input_shape[-1]
        if feature_dim % self.head_num != 0:
            raise IndexError('Invalid head number %d with the given input dim %d' % (self.head_num, feature_dim))
        for i in range(self.head_num):
            layer = keras.layers.Dense(
                units=feature_dim // self.head_num,
                activation=gelu,
                use_bias=False,
                trainable=self.trainable,
                name='%s-Dense-Q_%d' % (self.name, i + 1),
            )
            self.layers[layer.name] = layer
            layer = keras.layers.Dense(
                units=feature_dim // self.head_num,
                activation=gelu,
                use_bias=False,
                trainable=self.trainable,
                name='%s-Dense-K_%d' % (self.name, i + 1),
            )
            self.layers[layer.name] = layer
            layer = keras.layers.Dense(
                units=feature_dim // self.head_num,
                activation=gelu,
                use_bias=False,
                trainable=self.trainable,
                name='%s-Dense-V_%d' % (self.name, i + 1),
            )
            self.layers[layer.name] = layer
            layer = keras.layers.Dropout(
                rate=self.dropout_rate,
                trainable=self.trainable,
                name='%s-Dense-Dropout-Q_%d' % (self.name, i + 1),
            )
            self.layers[layer.name] = layer
            layer = keras.layers.Dropout(
                rate=self.dropout_rate,
                trainable=self.trainable,
                name='%s-Dense-Dropout-K_%d' % (self.name, i + 1),
            )
            self.layers[layer.name] = layer
            layer = keras.layers.Dropout(
                rate=self.dropout_rate,
                trainable=self.trainable,
                name='%s-Dense-Dropout-V_%d' % (self.name, i + 1),
            )
            self.layers[layer.name] = layer
            layer = Attention(
                trainable=self.trainable,
                name='%s-Attention_%d' % (self.name, i + 1),
            )
            self.layers[layer.name] = layer
            layer = keras.layers.Dropout(
                rate=self.dropout_rate,
                trainable=self.trainable,
                name='%s-Attention-Dropout_%d' % (self.name, i + 1),
            )
            self.layers[layer.name] = layer
        if self.head_num > 1:
            layer = keras.layers.Concatenate(name='%s-Concat' % self.name)
            self.layers[layer.name] = layer
        layer = keras.layers.Dense(
            units=feature_dim,
            activation=gelu,
            use_bias=False,
            trainable=self.trainable,
            name='%s-Dense_O' % self.name,
        )
        self.layers[layer.name] = layer
        layer = keras.layers.Dropout(
            rate=self.dropout_rate,
            trainable=self.trainable,
            name='%s-Dense-Dropout_O' % self.name,
        )
        self.layers[layer.name] = layer
        super(MultiHeadAttention, self).build(input_shape)

    def call(self, inputs, mask=None):
        outputs = []
        for i in range(self.head_num):
            query_layer = self.layers['%s-Dense-Q_%d' % (self.name, i + 1)](inputs)
            key_layer = self.layers['%s-Dense-K_%d' % (self.name, i + 1)](inputs)
            value_layer = self.layers['%s-Dense-V_%d' % (self.name, i + 1)](inputs)
            query_dropout_layer = self.layers['%s-Dense-Dropout-Q_%d' % (self.name, i + 1)](query_layer)
            key_dropout_layer = self.layers['%s-Dense-Dropout-K_%d' % (self.name, i + 1)](key_layer)
            value_dropout_layer = self.layers['%s-Dense-Dropout-V_%d' % (self.name, i + 1)](value_layer)
            att_layer = self.layers['%s-Attention_%d' % (self.name, i + 1)]([
                query_dropout_layer,
                key_dropout_layer,
                value_dropout_layer,
            ])
            dropout_layer = self.layers['%s-Attention-Dropout_%d' % (self.name, i + 1)](att_layer)
            outputs.append(dropout_layer)
        if self.head_num == 1:
            concat_layer = outputs[0]
        else:
            concat_layer = self.layers['%s-Concat' % self.name](outputs)
        dense_layer = self.layers['%s-Dense_O' % self.name](concat_layer)
        dropout_layer = self.layers['%s-Dense-Dropout_O' % self.name](dense_layer)
        return dropout_layer
