import random
import keras
import numpy as np
from keras_self_attention import ScaledDotProductAttention
from .layers import (get_inputs, Embeddings, Transformer, MultiHeadAttention,
                     FeedForward, Masked, Extract, LayerNormalization)
from .activations import gelu


TOKEN_PAD = ''  # Token for padding
TOKEN_UNK = '<UNK>'  # Token for unknown words
TOKEN_CLS = '<CLS>'  # Token for classification
TOKEN_SEP = '<SEP>'  # Token for separation
TOKEN_MASK = '<MASK>'  # Token for masking


def get_model(token_num,
              pos_num=512,
              seq_len=512,
              embed_dim=768,
              transformer_num=12,
              head_num=12,
              feed_forward_dim=3072,
              dropout_rate=0.1,
              custom_layers=None,
              training=True,
              lr=1e-4):
    """Get BERT model.

    See: https://arxiv.org/pdf/1810.04805.pdf

    :param token_num: Number of tokens.
    :param pos_num: Maximum position.
    :param seq_len: Maximum length of the input sequence or None.
    :param embed_dim: Dimensions of embeddings.
    :param transformer_num: Number of transformers.
    :param head_num: Number of heads in multi-head attention in each transformer.
    :param feed_forward_dim: Dimension of the feed forward layer in each transformer.
    :param dropout_rate: Dropout rate.
    :param custom_layers: A function that takes the embedding tensor and returns the tensor after feature extraction.
                          Arguments such as `transformer_num` and `head_num` will be ignored if `custom_layer` is not
                          `None`.
    :param training: The built model will be returned if it is `True`, otherwise the input layers and the last feature
                     extraction layer will be returned.
    :param lr: Learning rate.
    :return: The compiled model.
    """
    inputs = get_inputs(seq_len=seq_len)
    embed_layer = Embeddings(
        input_dim=token_num,
        output_dim=embed_dim,
        position_dim=pos_num,
        dropout_rate=dropout_rate,
        trainable=training,
        name='Embeddings',
    )(inputs[:3])
    transformed = embed_layer
    if custom_layers is not None:
        kwargs = {}
        if keras.utils.generic_utils.has_arg(custom_layers, 'trainable'):
            kwargs['trainable'] = training
        transformed = custom_layers(transformed, **kwargs)
    else:
        for i in range(transformer_num):
            transformed = Transformer(
                head_num=head_num,
                hidden_dim=feed_forward_dim,
                dropout_rate=dropout_rate,
                trainable=training,
                name='Transformer-%d' % (i + 1),
            )(transformed)
    if not training:
        return inputs, transformed
    mlm_pred_layer = keras.layers.Dense(
        units=token_num,
        activation='softmax',
        trainable=training,
        name='Dense-MLM',
    )(transformed)
    masked_layer = Masked(name='MLM')([mlm_pred_layer, inputs[-1]])
    extract_layer = Extract(index=0, name='Extract')(transformed)
    nsp_pred_layer = keras.layers.Dense(
        units=2,
        activation='softmax',
        trainable=training,
        name='NSP',
    )(extract_layer)
    model = keras.models.Model(inputs=inputs, outputs=[masked_layer, nsp_pred_layer])
    model.compile(
        optimizer=keras.optimizers.Adam(lr=lr),
        loss=keras.losses.sparse_categorical_crossentropy,
        metrics=[],
    )
    return model


def get_custom_objects():
    """Get all custom objects for loading saved models."""
    return {
        'Embeddings': Embeddings,
        'ScaledDotProductAttention': ScaledDotProductAttention,
        'MultiHeadAttention': MultiHeadAttention,
        'FeedForward': FeedForward,
        'LayerNormalization': LayerNormalization,
        'Transformer': Transformer,
        'Masked': Masked,
        'Extract': Extract,
        'gelu': gelu,
    }


def get_base_dict():
    """Get basic dictionary containing special tokens."""
    return {
        TOKEN_PAD: 0,
        TOKEN_UNK: 1,
        TOKEN_CLS: 2,
        TOKEN_SEP: 3,
        TOKEN_MASK: 4,
    }


def gen_batch_inputs(sentence_pairs,
                     token_dict,
                     token_list,
                     seq_len=512,
                     mask_rate=0.15,
                     mask_mask_rate=0.8,
                     mask_random_rate=0.1,
                     swap_sentence_rate=0.5,
                     force_mask=True):
    """Generate a batch of inputs and outputs for training.

    :param sentence_pairs: A list of pairs containing lists of tokens.
    :param token_dict: The dictionary containing special tokens.
    :param token_list: A list containing all tokens.
    :param seq_len: Length of the sequence.
    :param mask_rate: The rate of choosing a token for prediction.
    :param mask_mask_rate: The rate of replacing the token to `TOKEN_MASK`.
    :param mask_random_rate: The rate of replacing the token to a random word.
    :param swap_sentence_rate: The rate of swapping the second sentences.
    :param force_mask: At least one position will be masked.
    :return: All the inputs and outputs.
    """
    batch_size = len(sentence_pairs)
    base_dict = get_base_dict()
    unknown_index = token_dict[TOKEN_UNK]
    # Generate sentence swapping mapping
    nsp_outputs = np.ones((batch_size,))
    mapping = {}
    if swap_sentence_rate > 0.0:
        indices = [index for index in range(batch_size) if random.random() < swap_sentence_rate]
        mapped = indices[:]
        random.shuffle(mapped)
        for i in range(len(mapped)):
            if indices[i] != mapped[i]:
                nsp_outputs[indices[i]] = 0.0
        mapping = {indices[i]: mapped[i] for i in range(len(indices))}
    position_inputs = [[i for i in range(seq_len)] for _ in range(batch_size)]
    # Generate MLM
    token_inputs, segment_inputs, masked_inputs = [], [], []
    mlm_outputs = []
    for i in range(batch_size):
        first, second = sentence_pairs[i][0], sentence_pairs[mapping.get(i, i)][1]
        segment_inputs.append([0] * (len(first) + 2) + [1] * (seq_len - (len(first) + 2)))
        tokens = [TOKEN_CLS] + first + [TOKEN_SEP] + second + [TOKEN_SEP]
        tokens = tokens[:seq_len]
        tokens += [TOKEN_PAD] * (seq_len - len(tokens))
        token_input, masked_input, mlm_output = [], [], []
        has_mask = False
        for token in tokens:
            mlm_output.append(token_dict.get(token, unknown_index))
            if token not in base_dict and random.random() < mask_rate:
                has_mask = True
                masked_input.append(1)
                r = random.random()
                if r < mask_mask_rate:
                    token_input.append(token_dict[TOKEN_MASK])
                elif r < mask_mask_rate + mask_random_rate:
                    while True:
                        token = random.choice(token_list)
                        if token not in base_dict:
                            token_input.append(token_dict[token])
                            break
                else:
                    token_input.append(token_dict.get(token, unknown_index))
            else:
                masked_input.append(0)
                token_input.append(token_dict.get(token, unknown_index))
        if force_mask and not has_mask:
            masked_input[1] = 1
        token_inputs.append(token_input)
        masked_inputs.append(masked_input)
        mlm_outputs.append(mlm_output)
    inputs = [np.asarray(x) for x in [token_inputs, segment_inputs, position_inputs, masked_inputs]]
    outputs = [np.asarray(np.expand_dims(x, axis=-1)) for x in [mlm_outputs, nsp_outputs]]
    return inputs, outputs
