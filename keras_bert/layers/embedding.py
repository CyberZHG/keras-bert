import keras
from keras_layer_normalization import LayerNormalization


def get_embedding(inputs, token_num, pos_num, embed_dim, dropout_rate=0.1):
    """Get embedding layer.

    See: https://arxiv.org/pdf/1810.04805.pdf

    :param inputs: Input layers.
    :param token_num: Number of tokens.
    :param pos_num: Maximum position.
    :param embed_dim: The dimension of all embedding layers.
    :param dropout_rate: Dropout rate.
    :return: The merged embedding layer.
    """
    embeddings = [
        keras.layers.Embedding(
            input_dim=token_num,
            output_dim=embed_dim,
            mask_zero=True,
            name='Embedding-Token',
        )(inputs[0]),
        keras.layers.Embedding(
            input_dim=2,
            output_dim=embed_dim,
            name='Embedding-Segment',
        )(inputs[1]),
        keras.layers.Embedding(
            input_dim=pos_num,
            output_dim=embed_dim,
            name='Embedding-Position',
        )(inputs[2]),
    ]
    embed_layer = keras.layers.Add(name='Embedding')(embeddings)
    if dropout_rate > 0.0:
        dropout_layer = keras.layers.Dropout(
            rate=dropout_rate,
            name='Embedding-Dropout',
        )(embed_layer)
    else:
        dropout_layer = embed_layer
    norm_layer = LayerNormalization(name='Embedding-Norm')(dropout_layer)
    return norm_layer
