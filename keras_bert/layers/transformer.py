import keras
from .multi_head import get_multi_head_attention
from .layer_norm import LayerNormalization
from .feed_forward import FeedForward


def get_transformer(inputs, head_num, hidden_dim, name, dropout=0.1):
    """Generate a set of transformer layers.

    See: https://arxiv.org/pdf/1706.03762.pdf

    :param inputs: Input layer.
    :param head_num: Number of heads.
    :param hidden_dim: Hidden dimension for feed forward layer.
    :param name: Name for the transformer.
    :param dropout: Dropout rate.
    :return: Output layer.
    """
    multi_head_layer = get_multi_head_attention(
        inputs=inputs,
        head_num=head_num,
        dropout=dropout,
        name='%s-MultiHead' % name,
    )
    multi_head_norm = LayerNormalization(
        name='%s-MultiHead-Norm' % name,
    )(multi_head_layer)
    multi_head_dropout_layer = keras.layers.Dropout(
        rate=dropout,
        name='%s-MultiHead-Dropout' % name,
    )(multi_head_norm)
    multi_head_residual_layer = keras.layers.Add(
        name='%s-MultiHead-Add' % name,
    )([inputs, multi_head_dropout_layer])
    feed_forward_layer = FeedForward(
        hidden_dim=hidden_dim,
        name='%s-FeedForward' % name,
    )(multi_head_residual_layer)
    feed_forward_norm = LayerNormalization(
        name='%s-FeedForward-Norm' % name,
    )(feed_forward_layer)
    feed_forward_dropout_layer = keras.layers.Dropout(
        rate=dropout,
        name='%s-FeedForward-Dropout' % name,
    )(feed_forward_norm)
    feed_forward_residual_layer = keras.layers.Add(
        name='%s-FeedForward-Add' % name,
    )([multi_head_residual_layer, feed_forward_dropout_layer])
    return feed_forward_residual_layer
