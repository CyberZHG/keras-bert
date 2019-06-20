# coding=utf-8
from __future__ import unicode_literals

import os
from collections import namedtuple

import numpy as np

from .backend import keras
from .backend import backend as K
from .layers import Extract, MaskedGlobalMaxPool1D
from .loader import load_trained_model_from_checkpoint, load_vocabulary
from .tokenizer import Tokenizer

__all__ = [
    'POOL_NSP', 'POOL_MAX', 'POOL_AVE',
    'get_checkpoint_paths', 'extract_embeddings_generator', 'extract_embeddings',
]


POOL_NSP = 'POOL_NSP'
POOL_MAX = 'POOL_MAX'
POOL_AVE = 'POOL_AVE'


def get_checkpoint_paths(model_path):
    CheckpointPaths = namedtuple('CheckpointPaths', ['config', 'checkpoint', 'vocab'])
    config_path = os.path.join(model_path, 'bert_config.json')
    checkpoint_path = os.path.join(model_path, 'bert_model.ckpt')
    vocab_path = os.path.join(model_path, 'vocab.txt')
    return CheckpointPaths(config_path, checkpoint_path, vocab_path)


def extract_embeddings_generator(model,
                                 texts,
                                 poolings=None,
                                 vocabs=None,
                                 cased=False,
                                 batch_size=4,
                                 cut_embed=True,
                                 output_layer_num=1):
    """Extract embeddings from texts.

    :param model: Path to the checkpoint or built model without MLM and NSP.
    :param texts: Iterable texts.
    :param poolings: Pooling methods. Word embeddings will be returned if it is None.
                     Otherwise concatenated pooled embeddings will be returned.
    :param vocabs: A dict should be provided if model is built.
    :param cased: Whether it is cased for tokenizer.
    :param batch_size: Batch size.
    :param cut_embed: The computed embeddings will be cut based on their input lengths.
    :param output_layer_num: The number of layers whose outputs will be concatenated as a single output.
                             Only available when `model` is a path to checkpoint.
    :return: A list of numpy arrays representing the embeddings.
    """
    if isinstance(model, (str, type(u''))):
        paths = get_checkpoint_paths(model)
        model = load_trained_model_from_checkpoint(
            config_file=paths.config,
            checkpoint_file=paths.checkpoint,
            output_layer_num=output_layer_num,
        )
        vocabs = load_vocabulary(paths.vocab)

    seq_len = K.int_shape(model.outputs[0])[1]
    tokenizer = Tokenizer(vocabs, cased=cased)

    def _batch_generator():
        tokens, segments = [], []

        def _pad_inputs():
            if seq_len is None:
                max_len = max(map(len, tokens))
                for i in range(len(tokens)):
                    tokens[i].extend([0] * (max_len - len(tokens[i])))
                    segments[i].extend([0] * (max_len - len(segments[i])))
            return [np.array(tokens), np.array(segments)]

        for text in texts:
            if isinstance(text, (str, type(u''))):
                token, segment = tokenizer.encode(text, max_len=seq_len)
            else:
                token, segment = tokenizer.encode(text[0], text[1], max_len=seq_len)
            tokens.append(token)
            segments.append(segment)
            if len(tokens) == batch_size:
                yield _pad_inputs()
                tokens, segments = [], []
        if len(tokens) > 0:
            yield _pad_inputs()

    if poolings is not None:
        if isinstance(poolings, (str, type(u''))):
            poolings = [poolings]
        outputs = []
        for pooling in poolings:
            if pooling == POOL_NSP:
                outputs.append(Extract(index=0, name='Pool-NSP')(model.outputs[0]))
            elif pooling == POOL_MAX:
                outputs.append(MaskedGlobalMaxPool1D(name='Pool-Max')(model.outputs[0]))
            elif pooling == POOL_AVE:
                outputs.append(keras.layers.GlobalAvgPool1D(name='Pool-Ave')(model.outputs[0]))
            else:
                raise ValueError('Unknown pooling method: {}'.format(pooling))
        if len(outputs) == 1:
            outputs = outputs[0]
        else:
            outputs = keras.layers.Concatenate(name='Concatenate')(outputs)
        model = keras.models.Model(inputs=model.inputs, outputs=outputs)

    for batch_inputs in _batch_generator():
        outputs = model.predict(batch_inputs)
        for inputs, output in zip(batch_inputs[0], outputs):
            if poolings is None and cut_embed:
                length = 0
                for i in range(len(inputs) - 1, -1, -1):
                    if inputs[i] != 0:
                        length = i + 1
                        break
                output = output[:length]
            yield output


def extract_embeddings(model,
                       texts,
                       poolings=None,
                       vocabs=None,
                       cased=False,
                       batch_size=4,
                       cut_embed=True,
                       output_layer_num=1):
    """Extract embeddings from texts.

    :param model: Path to the checkpoint or built model without MLM and NSP.
    :param texts: Iterable texts.
    :param poolings: Pooling methods. Word embeddings will be returned if it is None.
                     Otherwise concatenated pooled embeddings will be returned.
    :param vocabs: A dict should be provided if model is built.
    :param cased: Whether it is cased for tokenizer.
    :param batch_size: Batch size.
    :param cut_embed: The computed embeddings will be cut based on their input lengths.
    :param output_layer_num: The number of layers whose outputs will be concatenated as a single output.
                             Only available when `model` is a path to checkpoint.
    :return: A list of numpy arrays representing the embeddings.
    """
    return [embedding for embedding in extract_embeddings_generator(
        model, texts, poolings, vocabs, cased, batch_size, cut_embed, output_layer_num
    )]
