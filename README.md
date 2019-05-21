# Keras BERT

[![Travis](https://travis-ci.org/CyberZHG/keras-bert.svg)](https://travis-ci.org/CyberZHG/keras-bert)
[![Coverage](https://coveralls.io/repos/github/CyberZHG/keras-bert/badge.svg?branch=master)](https://coveralls.io/github/CyberZHG/keras-bert)
[![Version](https://img.shields.io/pypi/v/keras-bert.svg)](https://pypi.org/project/keras-bert/)

Implementation of the [BERT](https://arxiv.org/pdf/1810.04805.pdf). Official pre-trained models could be loaded for feature extraction and prediction.

## Install

```bash
pip install keras-bert
```

## Usage

### Load Official Pre-trained Models

In [feature extraction demo](./demo/load_model/load_and_extract.py), you should be able to get the same extraction results as the official model `chinese_L-12_H-768_A-12`. And in [prediction demo](./demo/load_model/load_and_predict.py), the missing word in the sentence could be predicted.


### Run on TPU

[Colab demo](https://colab.research.google.com/github/CyberZHG/keras-bert/blob/master/demo/load_model/keras_bert_load_and_extract_tpu.ipynb) shows how to convert to a model that runs on TPU.

### Tokenizer

The `Tokenizer` class is used for splitting texts and generating indices:

```python
from keras_bert import Tokenizer

token_dict = {
    '[CLS]': 0,
    '[SEP]': 1,
    'un': 2,
    '##aff': 3,
    '##able': 4,
    '[UNK]': 5,
}
tokenizer = Tokenizer(token_dict)
print(tokenizer.tokenize('unaffable'))  # The result should be `['[CLS]', 'un', '##aff', '##able', '[SEP]']`
indices, segments = tokenizer.encode('unaffable')
print(indices)  # Should be `[0, 2, 3, 4, 1]`
print(segments)  # Should be `[0, 0, 0, 0, 0]`

print(tokenizer.tokenize(first='unaffable', second='钢'))
# The result should be `['[CLS]', 'un', '##aff', '##able', '[SEP]', '钢', '[SEP]']`
indices, segments = tokenizer.encode(first='unaffable', second='钢', max_len=10)
print(indices)  # Should be `[0, 2, 3, 4, 1, 5, 1, 0, 0, 0]`
print(segments)  # Should be `[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]`
```

### Train & Use

```python
import keras
from keras_bert import get_base_dict, get_model, gen_batch_inputs


# A toy input example
sentence_pairs = [
    [['all', 'work', 'and', 'no', 'play'], ['makes', 'jack', 'a', 'dull', 'boy']],
    [['from', 'the', 'day', 'forth'], ['my', 'arm', 'changed']],
    [['and', 'a', 'voice', 'echoed'], ['power', 'give', 'me', 'more', 'power']],
]


# Build token dictionary
token_dict = get_base_dict()  # A dict that contains some special tokens
for pairs in sentence_pairs:
    for token in pairs[0] + pairs[1]:
        if token not in token_dict:
            token_dict[token] = len(token_dict)
token_list = list(token_dict.keys())  # Used for selecting a random word


# Build & train the model
model = get_model(
    token_num=len(token_dict),
    head_num=5,
    transformer_num=12,
    embed_dim=25,
    feed_forward_dim=100,
    seq_len=20,
    pos_num=20,
    dropout_rate=0.05,
)
model.summary()

def _generator():
    while True:
        yield gen_batch_inputs(
            sentence_pairs,
            token_dict,
            token_list,
            seq_len=20,
            mask_rate=0.3,
            swap_sentence_rate=1.0,
        )

model.fit_generator(
    generator=_generator(),
    steps_per_epoch=1000,
    epochs=100,
    validation_data=_generator(),
    validation_steps=100,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    ],
)


# Use the trained model
inputs, output_layer = get_model(
    token_num=len(token_dict),
    head_num=5,
    transformer_num=12,
    embed_dim=25,
    feed_forward_dim=100,
    seq_len=20,
    pos_num=20,
    dropout_rate=0.05,
    training=False,      # The input layers and output layer will be returned if `training` is `False`
    trainable=False,     # Whether the model is trainable. The default value is the same with `training`
    output_layer_num=4,  # The number of layers whose outputs will be concatenated as a single output.
                         # Only available when `training` is `False`.
)
```

### Use `tensorflow.python.keras`

Add `TF_KERAS=1` to environment variables to use `tensorflow.python.keras`.

### Use `theano` Backend

Add `KERAS_BACKEND=theano` to environment variables to enable `theano` backend.
