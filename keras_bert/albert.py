from keras_pos_embd import PositionEmbedding
from keras_layer_normalization import LayerNormalization
from keras_multi_head import MultiHeadAttention
from keras_position_wise_feed_forward import FeedForward
from .backend import keras
from .activations import gelu
from .layers import TokenEmbedding, EmbeddingSimilarity, Masked, Extract


__all__ = [
    'TOKEN_PAD', 'TOKEN_UNK', 'TOKEN_CLS', 'TOKEN_SEP', 'TOKEN_MASK', 'get_model',
]


TOKEN_PAD = '<pad>'  # Token for padding
TOKEN_UNK = '<unk>'  # Token for unknown words
TOKEN_CLS = '[CLS]'  # Token for classification
TOKEN_SEP = '[SEP]'  # Token for separation
TOKEN_MASK = '[MASK]'  # Token for masking


def get_model(vocab_size,
              type_vocab_size=2,
              max_position_embeddings=512,
              embedding_size=128,
              hidden_size=768,
              hidden_dropout_prob=0.0,
              num_hidden_layers=12,
              num_attention_heads=12,
              attention_probs_dropout_prob=0.0,
              hidden_act='gelu',
              intermediate_size=3072,
              training=True,
              trainable=True,
              seq_len=None,
              output_layers=None):
    """Get ALBERT model.

    See: https://arxiv.org/abs/1909.11942

    :param vocab_size: Number of tokens.
    :param type_vocab_size: Number of types.
    :param max_position_embeddings: Maximum position.
    :param embedding_size: Dimensions of embeddings.
    :param hidden_size: Dimensions of transformer blocks.
    :param hidden_dropout_prob: Dropout rate.
    :param num_hidden_layers: Number of transformer blocks.
    :param num_attention_heads: Number of heads in multi-head attention in each transformer.
    :param attention_probs_dropout_prob: Dropout rate.
    :param hidden_act: Activation for attention layers.
    :param intermediate_size: Dimension of the feed forward layer in each transformer.
    :param training: A built model with MLM and SOP outputs will be returned if it is `True`,
                     otherwise the input layers and the last feature extraction layer will be returned.
    :param trainable: Whether the model is trainable.
    :param seq_len: The length of input sequence.
    :param output_layers: A list contains the indices of layers to be concatenated.
                          Only available when `training` is `False`.
    :return: The built model.
    """
    if hidden_act == 'gelu':
        hidden_act = gelu

    input_token = keras.layers.Input(shape=(seq_len,), name='Input-Token')
    input_type = keras.layers.Input(shape=(seq_len,), name='Input-Type')
    inputs = [input_token, input_type]
    if training:
        input_mask = keras.layers.Input(shape=(seq_len,), name='Input-Mask')
        inputs.append(input_mask)

    embed_token, embed_weights = TokenEmbedding(
        input_dim=vocab_size,
        output_dim=embedding_size,
        mask_zero=True,
        trainable=trainable,
        name='Embedding-Token',
    )(input_token)
    embed_type = keras.layers.Embedding(
        input_dim=type_vocab_size,
        output_dim=embedding_size,
        mask_zero=False,
        trainable=trainable,
        name='Embedding-Type',
    )(input_type)
    embed_token_and_type = keras.layers.Add(
        name='Embedding-Token-Type',
    )([embed_token, embed_type])
    embed_position = PositionEmbedding(
        input_dim=max_position_embeddings,
        output_dim=embedding_size,
        mode=PositionEmbedding.MODE_ADD,
        mask_zero=False,
        trainable=trainable,
        name='Embedding-Position',
    )(embed_token_and_type)

    embed_normal = LayerNormalization(
        trainable=trainable,
        name='Embedding-Normal',
    )(embed_position)
    embed_dropout = embed_normal
    if hidden_dropout_prob > 0.0:
        embed_dropout = keras.layers.Dropout(
            rate=hidden_dropout_prob,
            name='Embedding-Dropout',
        )(embed_normal)
    hidden = keras.layers.Dense(
        units=hidden_size,
        name='Hidden',
    )(embed_dropout)

    attention_layer = MultiHeadAttention(
        head_num=num_attention_heads,
        activation=None,
        name='Attention',
    )
    attention_normal = LayerNormalization(name='Attention-Normal')
    feed_forward_layer = FeedForward(
        units=intermediate_size,
        activation=hidden_act,
        name='Feed-Forward'
    )
    feed_forward_normal = LayerNormalization(name='Feed-Forward-Normal')

    hidden_outputs = []
    for i in range(num_hidden_layers):
        attention_input = hidden
        hidden = attention_layer(attention_input)
        if hidden_dropout_prob > 0.0:
            hidden = keras.layers.Dropout(
                rate=hidden_dropout_prob,
                name='Attention-Dropout-{}'.format(i + 1),
            )(hidden)
        hidden = keras.layers.Add(name='Attention-Add-{}'.format(i + 1))([attention_input, hidden])
        hidden = attention_normal(hidden)

        feed_forward_input = hidden
        hidden = feed_forward_layer(hidden)
        if hidden_dropout_prob > 0.0:
            hidden = keras.layers.Dropout(
                rate=hidden_dropout_prob,
                name='Feed-Forward-Dropout-{}'.format(i + 1),
            )(hidden)
        hidden = keras.layers.Add(
            name='Feed-Forward-Add-{}'.format(i + 1),
        )([feed_forward_input, hidden])
        hidden = feed_forward_normal(hidden)

        hidden_outputs.append(hidden)

    if training:
        mlm_dense = keras.layers.Dense(
            units=embedding_size,
            activation='tanh',
            name='MLM-Dense')(hidden)
        mlm_normal = LayerNormalization(name='MLM-Normal')(mlm_dense)
        mlm_predict = EmbeddingSimilarity(name='MLM-Sim')([mlm_normal, embed_weights])
        mlm = Masked(name='MLM')([mlm_predict, inputs[-1]])

        extracted = Extract(index=0, name='Extract')(hidden)
        sop_dense = keras.layers.Dense(
            units=hidden_size,
            activation='tanh',
            name='NSP-Dense',
        )(extracted)
        sop = keras.layers.Dense(
            units=2,
            activation='softmax',
            name='NSP',
        )(sop_dense)
        model = keras.models.Model(inputs=inputs, outputs=[mlm, sop])
    else:
        if isinstance(output_layers, int):
            hidden = hidden_outputs[output_layers]
        elif output_layers is not None:
            hidden = keras.layers.Concatenate(name='Output')([hidden_outputs[index] for index in output_layers])
        model = keras.models.Model(inputs=inputs, outputs=hidden)
    return model
