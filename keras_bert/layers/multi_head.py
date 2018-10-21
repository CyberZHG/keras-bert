import keras
from .attention import Attention
from keras_bert.activations.gelu import gelu


def get_multi_head_attention(inputs, head_num, name, dropout=0.1):
    """Generate multi-head attention layers.

    See: https://arxiv.org/pdf/1706.03762.pdf

    :param inputs: Input layer.
    :param head_num: Number of heads.
    :param name: Name for multi-head.
    :param dropout: Dropout rate.
    :return: Output layer.
    """
    input_dim = inputs.get_shape().as_list()[-1]
    assert input_dim % head_num == 0
    outputs = []
    for i in range(head_num):
        dense_layer = keras.layers.Dense(
            units=input_dim // head_num,
            activation=gelu,
            use_bias=False,
            name='%s-Dense_%d' % (name, i + 1),
        )(inputs)
        dropout_layer = keras.layers.Dropout(
            rate=dropout,
            name='%s-Dense-Dropout_%d' % (name, i + 1),
        )(dense_layer)
        att_layer = Attention(
            name='%s-Attention_%d' % (name, i + 1),
        )(dropout_layer)
        dropout_layer = keras.layers.Dropout(
            rate=dropout,
            name='%s-Attention_-Dropout_%d' % (name, i + 1),
        )(att_layer)
        outputs.append(dropout_layer)
    if len(outputs) == 1:
        concat_layer = outputs[0]
    else:
        concat_layer = keras.layers.Concatenate(name='%s-Concat' % name)(outputs)
    dense_layer = keras.layers.Dense(
        units=input_dim,
        activation=gelu,
        use_bias=False,
        name='%s-Dense_O' % name,
    )(concat_layer)
    dropout_layer = keras.layers.Dropout(
        rate=dropout,
        name='%s-Dense-Dropout_O' % name,
    )(dense_layer)
    return dropout_layer
