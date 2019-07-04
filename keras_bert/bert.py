import math

import numpy as np
from keras_pos_embd import PositionEmbedding
from keras_layer_normalization import LayerNormalization
from keras_transformer import get_encoders
from keras_transformer import get_custom_objects as get_encoder_custom_objects
from .backend import keras
from .backend import backend as K
from .layers import get_inputs, get_embedding, TokenEmbedding, EmbeddingSimilarity, Masked, Extract
from .optimizers import AdamWarmup


__all__ = [
    'TOKEN_PAD', 'TOKEN_UNK', 'TOKEN_CLS', 'TOKEN_SEP', 'TOKEN_MASK',
    'gelu', 'get_model', 'get_base_dict', 'gen_batch_inputs', 'get_token_embedding',
    'get_custom_objects', 'set_custom_objects',
]


TOKEN_PAD = ''  # Token for padding
TOKEN_UNK = '[UNK]'  # Token for unknown words
TOKEN_CLS = '[CLS]'  # Token for classification
TOKEN_SEP = '[SEP]'  # Token for separation
TOKEN_MASK = '[MASK]'  # Token for masking


def gelu_tensorflow(x):
    from tensorflow.python.ops.math_ops import erf, sqrt
    return 0.5 * x * (1.0 + erf(x / sqrt(2.0)))


def gelu_fallback(x):
    return 0.5 * x * (1.0 + K.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * K.pow(x, 3))))


if K.backend() == 'tensorflow':
    gelu = gelu_tensorflow
else:
    gelu = gelu_fallback


def get_model(token_num,
              pos_num=512,
              seq_len=512,
              embed_dim=768,
              transformer_num=12,
              head_num=12,
              feed_forward_dim=3072,
              dropout_rate=0.1,
              weight_decay=0.01,
              attention_activation=None,
              feed_forward_activation='gelu',
              training=True,
              trainable=None,
              output_layer_num=1,
              decay_steps=100000,
              warmup_steps=10000,
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
    :param weight_decay: Weight decay rate.
    :param attention_activation: Activation for attention layers.
    :param feed_forward_activation: Activation for feed-forward layers.
    :param training: A built model with MLM and NSP outputs will be returned if it is `True`,
                     otherwise the input layers and the last feature extraction layer will be returned.
    :param trainable: Whether the model is trainable.
    :param output_layer_num: The number of layers whose outputs will be concatenated as a single output.
                             Only available when `training` is `False`.
    :param decay_steps: Learning rate will decay linearly to zero in decay steps.
    :param warmup_steps: Learning rate will increase linearly to lr in first warmup steps.
    :param lr: Learning rate.
    :return: The compiled model.
    """
    if attention_activation == 'gelu':
        attention_activation = gelu
    if feed_forward_activation == 'gelu':
        feed_forward_activation = gelu
    if trainable is None:
        trainable = training

    def _trainable(_layer):
        if isinstance(trainable, (list, tuple, set)):
            for prefix in trainable:
                if _layer.name.startswith(prefix):
                    return True
            return False
        return trainable

    inputs = get_inputs(seq_len=seq_len)
    embed_layer, embed_weights = get_embedding(
        inputs,
        token_num=token_num,
        embed_dim=embed_dim,
        pos_num=pos_num,
        dropout_rate=dropout_rate,
    )
    transformed = get_encoders(
        encoder_num=transformer_num,
        input_layer=embed_layer,
        head_num=head_num,
        hidden_dim=feed_forward_dim,
        attention_activation=attention_activation,
        feed_forward_activation=feed_forward_activation,
        dropout_rate=dropout_rate,
    )
    if training:
        mlm_dense_layer = keras.layers.Dense(
            units=embed_dim,
            activation=feed_forward_activation,
            name='MLM-Dense',
        )(transformed)
        mlm_norm_layer = LayerNormalization(name='MLM-Norm')(mlm_dense_layer)
        mlm_pred_layer = EmbeddingSimilarity(name='MLM-Sim')([mlm_norm_layer, embed_weights])
        masked_layer = Masked(name='MLM')([mlm_pred_layer, inputs[-1]])
        extract_layer = Extract(index=0, name='Extract')(transformed)
        nsp_dense_layer = keras.layers.Dense(
            units=embed_dim,
            activation='tanh',
            name='NSP-Dense',
        )(extract_layer)
        nsp_pred_layer = keras.layers.Dense(
            units=2,
            activation='softmax',
            name='NSP',
        )(nsp_dense_layer)
        model = keras.models.Model(inputs=inputs, outputs=[masked_layer, nsp_pred_layer])
        for layer in model.layers:
            layer.trainable = _trainable(layer)
        model.compile(
            optimizer=AdamWarmup(
                decay_steps=decay_steps,
                warmup_steps=warmup_steps,
                lr=lr,
                weight_decay=weight_decay,
                weight_decay_pattern=['embeddings', 'kernel', 'W1', 'W2', 'Wk', 'Wq', 'Wv', 'Wo'],
            ),
            loss=keras.losses.sparse_categorical_crossentropy,
        )
        return model
    else:
        inputs = inputs[:2]
        model = keras.models.Model(inputs=inputs, outputs=transformed)
        for layer in model.layers:
            layer.trainable = _trainable(layer)
        output_layer_num = min(output_layer_num, transformer_num)
        if output_layer_num > 1:
            outputs = []
            for i in range(output_layer_num):
                layer = model.get_layer(name='Encoder-{}-FeedForward-Norm'.format(transformer_num - i))
                outputs.append(layer.output)
            transformed = keras.layers.Concatenate(name='Encoder-Output')(list(reversed(outputs)))
        return inputs, transformed


def get_custom_objects():
    """Get all custom objects for loading saved models."""
    custom_objects = get_encoder_custom_objects()
    custom_objects['PositionEmbedding'] = PositionEmbedding
    custom_objects['TokenEmbedding'] = TokenEmbedding
    custom_objects['EmbeddingSimilarity'] = EmbeddingSimilarity
    custom_objects['Masked'] = Masked
    custom_objects['Extract'] = Extract
    custom_objects['gelu'] = gelu
    custom_objects['gelu_tensorflow'] = gelu_tensorflow
    custom_objects['gelu_fallback'] = gelu_fallback
    custom_objects['AdamWarmup'] = AdamWarmup
    return custom_objects


def set_custom_objects():
    """Add custom objects to Keras environments."""
    for k, v in get_custom_objects().items():
        keras.utils.get_custom_objects()[k] = v


def get_base_dict():
    """Get basic dictionary containing special tokens."""
    return {
        TOKEN_PAD: 0,
        TOKEN_UNK: 1,
        TOKEN_CLS: 2,
        TOKEN_SEP: 3,
        TOKEN_MASK: 4,
    }


def get_token_embedding(model):
    """Get token embedding from model.

    :param model: The built model.
    :return: The output weights of embeddings.
    """
    return model.get_layer('Embedding-Token').output[1]


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
    nsp_outputs = np.zeros((batch_size,))
    mapping = {}
    if swap_sentence_rate > 0.0:
        indices = [index for index in range(batch_size) if np.random.random() < swap_sentence_rate]
        mapped = indices[:]
        np.random.shuffle(mapped)
        for i in range(len(mapped)):
            if indices[i] != mapped[i]:
                nsp_outputs[indices[i]] = 1.0
        mapping = {indices[i]: mapped[i] for i in range(len(indices))}
    # Generate MLM
    token_inputs, segment_inputs, masked_inputs = [], [], []
    mlm_outputs = []
    for i in range(batch_size):
        first, second = sentence_pairs[i][0], sentence_pairs[mapping.get(i, i)][1]
        segment_inputs.append(([0] * (len(first) + 2) + [1] * (seq_len - (len(first) + 2)))[:seq_len])
        tokens = [TOKEN_CLS] + first + [TOKEN_SEP] + second + [TOKEN_SEP]
        tokens = tokens[:seq_len]
        tokens += [TOKEN_PAD] * (seq_len - len(tokens))
        token_input, masked_input, mlm_output = [], [], []
        has_mask = False
        for token in tokens:
            mlm_output.append(token_dict.get(token, unknown_index))
            if token not in base_dict and np.random.random() < mask_rate:
                has_mask = True
                masked_input.append(1)
                r = np.random.random()
                if r < mask_mask_rate:
                    token_input.append(token_dict[TOKEN_MASK])
                elif r < mask_mask_rate + mask_random_rate:
                    while True:
                        token = np.random.choice(token_list)
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
    inputs = [np.asarray(x) for x in [token_inputs, segment_inputs, masked_inputs]]
    outputs = [np.asarray(np.expand_dims(x, axis=-1)) for x in [mlm_outputs, nsp_outputs]]
    return inputs, outputs
