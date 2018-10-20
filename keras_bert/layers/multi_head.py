import keras
from .attention import Attention


def get_multi_head_attention(inputs, head_num, name):
    """Generate multi-head attention layers.

    :param inputs: Input layer.
    :param head_num: Number of heads.
    :param name: Name for multi-head.
    :return: Output layer.
    """
    input_dim = inputs.get_shape().as_list()[-1]
    assert input_dim % head_num == 0
    outputs = []
    for i in range(head_num):
        dense_layer = keras.layers.Dense(
            units=input_dim // head_num,
            use_bias=False,
            name='%s-Dense_%d' % (name, i + 1),
        )(inputs)
        att_layer = Attention(name='%s-Attention_%d' % (name, i + 1))(dense_layer)
        outputs.append(att_layer)
    concat_layer = keras.layers.Concatenate(name='%s-Concat' % name)(outputs)
    dense_layer = keras.layers.Dense(
        units=input_dim,
        use_bias=False,
        name='%s-Dense_O' % name,
    )(concat_layer)
    return dense_layer
