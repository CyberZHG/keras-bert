import copy
import keras


def get_multi_head(inputs, layer, head_num, hidden_dim, name):
    """Generate multi-head layers.

    :param inputs: Input layer.
    :param layer: Layers to be duplicated.
    :param head_num: Number of heads.
    :param hidden_dim: Input dimension for each duplicated layer.
    :param name: Name for multi-head.
    :return: Output layer.
    """
    outputs = []
    for i in range(head_num):
        dense_layer = keras.layers.Dense(
            units=hidden_dim,
            name='%s-Dense_%d' % (name, i + 1),
        )(inputs)
        dup_layer = copy.deepcopy(layer)
        dup_layer.name = dup_layer.name + '_%d' % (i + 1)
        outputs.append(dup_layer(dense_layer))
    return keras.layers.Concatenate(name='%s-Concat' % name)(outputs)
