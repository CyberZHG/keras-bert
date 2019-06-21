# Keras BERT

[![Travis](https://travis-ci.org/CyberZHG/keras-bert.svg)](https://travis-ci.org/CyberZHG/keras-bert)
[![Coverage](https://coveralls.io/repos/github/CyberZHG/keras-bert/badge.svg?branch=master)](https://coveralls.io/github/CyberZHG/keras-bert)
[![Version](https://img.shields.io/pypi/v/keras-bert.svg)](https://pypi.org/project/keras-bert/)
![Downloads](https://img.shields.io/pypi/dm/keras-bert.svg)
![License](https://img.shields.io/pypi/l/keras-bert.svg)

![](https://img.shields.io/badge/keras-tensorflow-blue.svg)
![](https://img.shields.io/badge/keras-theano-blue.svg)
![](https://img.shields.io/badge/keras-tf.keras-blue.svg)
![](https://img.shields.io/badge/keras-tf.keras/eager-blue.svg)
![](https://img.shields.io/badge/keras-tf.keras/2.0_beta-blue.svg)

\[[中文](https://github.com/CyberZHG/keras-bert/blob/master/README.zh-CN.md)|[English](https://github.com/CyberZHG/keras-bert/blob/master/README.md)\]

Implementation of the [BERT](https://arxiv.org/pdf/1810.04805.pdf). Official pre-trained models could be loaded for feature extraction and prediction.

## Install

```bash
pip install keras-bert
```

## Usage

### Load Official Pre-trained Models

In [feature extraction demo](./demo/load_model/load_and_extract.py), you should be able to get the same extraction results as the official model `chinese_L-12_H-768_A-12`. And in [prediction demo](./demo/load_model/load_and_predict.py), the missing word in the sentence could be predicted.


### Run on TPU

The [extraction demo](https://colab.research.google.com/github/CyberZHG/keras-bert/blob/master/demo/load_model/keras_bert_load_and_extract_tpu.ipynb) shows how to convert to a model that runs on TPU.

The [classification demo](https://colab.research.google.com/github/CyberZHG/keras-bert/blob/master/demo/tune/keras_bert_classification_tpu.ipynb) shows how to apply the model to simple classification tasks.

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

### Use Warmup

`AdamWarmup` optimizer is provided for warmup and decay. The learning rate will reach `lr` in `warmpup_steps` steps, and decay to `min_lr` in `decay_steps` steps. There is a helper function `calc_train_steps` for calculating the two steps:

```python
import numpy as np
from keras_bert import AdamWarmup, calc_train_steps

train_x = np.random.standard_normal((1024, 100))

total_steps, warmup_steps = calc_train_steps(
    num_example=train_x.shape[0],
    batch_size=32,
    epochs=10,
    warmup_proportion=0.1,
)

optimizer = AdamWarmup(total_steps, warmup_steps, lr=1e-3, min_lr=1e-5)
```

### Download Pretrained Checkpoints

Several download urls has been added. You can get the downloaded and uncompressed path of a checkpoint by:

```python
from keras_bert import get_pretrained, PretrainedList, get_checkpoint_paths

model_path = get_pretrained(PretrainedList.multi_cased_base)
paths = get_checkpoint_paths(model_path)
print(paths.config, paths.checkpoint, paths.vocab)
```

### Extract Features

You can use helper function `extract_embeddings` if the features of tokens or sentences (without further tuning) are what you need. To extract the features of all tokens:

```python
from keras_bert import extract_embeddings

model_path = 'xxx/yyy/uncased_L-12_H-768_A-12'
texts = ['all work and no play', 'makes jack a dull boy~']

embeddings = extract_embeddings(model_path, texts)
```

The returned result is a list with the same length as texts. Each item in the list is a numpy array truncated by the length of the input. The shapes of outputs in this example are `(8, 768)` and `(9, 768)`.

When the inputs are paired-sentences, and you need the outputs of `NSP` and max-pooling of the last 4 layers:

```python
from keras_bert import extract_embeddings, POOL_NSP, POOL_MAX

model_path = 'xxx/yyy/uncased_L-12_H-768_A-12'
texts = [
    ('all work and no play', 'makes jack a dull boy'),
    ('makes jack a dull boy', 'all work and no play'),
]

embeddings = extract_embeddings(model_path, texts, output_layer_num=4, poolings=[POOL_NSP, POOL_MAX])
```

There are no token features in the results. The outputs of `NSP` and max-pooling will be concatenated with the final shape `(768 x 4 x 2,)`.

The second argument in the helper function is a generator. To extract features from file:

```python
import codecs
from keras_bert import extract_embeddings

model_path = 'xxx/yyy/uncased_L-12_H-768_A-12'

with codecs.open('xxx.txt', 'r', 'utf8') as reader:
    texts = map(lambda x: x.strip(), reader)
    embeddings = extract_embeddings(model_path, texts)
```

### Use `tensorflow.python.keras`

Add `TF_KERAS=1` to environment variables to use `tensorflow.python.keras`.

### Use `theano` Backend

Add `KERAS_BACKEND=theano` to environment variables to enable `theano` backend.
