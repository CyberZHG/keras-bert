import keras
from .multi_head import get_multi_head_attention
from .feed_forward import FeedForward


def get_transformer(inputs, head_num, hidden_dim, name, dropout=0.1):
    """Generate a set of transformer layers.

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
        name='%s-MultiHead' % name,
    )
    # TODO: Layer normalization
    multi_head_dropout_layer = keras.layers.Dropout(
        rate=dropout,
        name='%s-MultiHead-Dropout' % name,
    )(multi_head_layer)
    multi_head_residual_layer = keras.layers.Add(
        name='%s-MultiHead-Add' % name,
    )([inputs, multi_head_dropout_layer])
    feed_forward_layer = FeedForward(
        hidden_dim=hidden_dim,
        name='%s-FeedForward' % name,
    )(multi_head_residual_layer)
    # TODO: Layer normalization
    feed_forward_dropout_layer = keras.layers.Dropout(
        rate=dropout,
        name='%s-FeedForward-Dropout' % name,
    )(feed_forward_layer)
    feed_forward_residual_layer = keras.layers.Add(
        name='%s-FeedForward-Add' % name,
    )([multi_head_residual_layer, feed_forward_dropout_layer])
    return feed_forward_residual_layer
