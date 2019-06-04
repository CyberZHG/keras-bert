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

\[[中文](https://github.com/CyberZHG/keras-bert/blob/master/README.zh-CN.md)|[English](https://github.com/CyberZHG/keras-bert/blob/master/README.md)\]

[BERT](https://arxiv.org/pdf/1810.04805.pdf)的非官方实现，可以加载官方的预训练模型进行特征提取和预测。

## 安装

```bash
pip install keras-bert
```

## 使用

### 官方模型使用

[特征提取展示](./demo/load_model/load_and_extract.py)中使用官方预训练好的`chinese_L-12_H-768_A-12`可以得到和官方工具一样的结果。

[预测展示](./demo/load_model/load_and_predict.py)中可以填补出缺失词并预测是否是上下文。

### 使用TPU

[特征提取示例](https://colab.research.google.com/github/CyberZHG/keras-bert/blob/master/demo/load_model/keras_bert_load_and_extract_tpu.ipynb)中展示了如何在TPU上进行特征提取。

[分类示例](https://colab.research.google.com/github/CyberZHG/keras-bert/blob/master/demo/tune/keras_bert_classification_tpu.ipynb)中在IMDB数据集上对模型进行了微调以适应新的分类任务。

### 分词

`Tokenizer`类可以用来进行分词工作，包括归一化和英文部分的最大贪心匹配等，在CJK字符集内的中文会以单字分隔。

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
print(tokenizer.tokenize('unaffable'))  # 分词结果是：`['[CLS]', 'un', '##aff', '##able', '[SEP]']`

indices, segments = tokenizer.encode('unaffable')
print(indices)   # 词对应的下标：`[0, 2, 3, 4, 1]`
print(segments)  # 段落对应下标：`[0, 0, 0, 0, 0]`

print(tokenizer.tokenize(first='unaffable', second='钢'))
# 分词结果是：`['[CLS]', 'un', '##aff', '##able', '[SEP]', '钢', '[SEP]']`
indices, segments = tokenizer.encode(first='unaffable', second='钢', max_len=10)
print(indices)   # 词对应的下标：`[0, 2, 3, 4, 1, 5, 1, 0, 0, 0]`
print(segments)  # 段落对应下标：`[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]`
```

### 训练和使用

训练过程推荐使用官方的代码。这个代码库内包含一个的训练过程，`training`为`True`的情况下使用的是带warmup的Adam优化器：

```python
import keras
from keras_bert import get_base_dict, get_model, gen_batch_inputs


# 随便的输入样例：
sentence_pairs = [
    [['all', 'work', 'and', 'no', 'play'], ['makes', 'jack', 'a', 'dull', 'boy']],
    [['from', 'the', 'day', 'forth'], ['my', 'arm', 'changed']],
    [['and', 'a', 'voice', 'echoed'], ['power', 'give', 'me', 'more', 'power']],
]


# 构建自定义词典
token_dict = get_base_dict()  # 初始化特殊符号，如`[CLS]`
for pairs in sentence_pairs:
    for token in pairs[0] + pairs[1]:
        if token not in token_dict:
            token_dict[token] = len(token_dict)
token_list = list(token_dict.keys())  # Used for selecting a random word


# 构建和训练模型
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


# 使用训练好的模型
inputs, output_layer = get_model(
    token_num=len(token_dict),
    head_num=5,
    transformer_num=12,
    embed_dim=25,
    feed_forward_dim=100,
    seq_len=20,
    pos_num=20,
    dropout_rate=0.05,
    training=False,      # 当`training`是`False`，返回值是输入和输出
    trainable=False,     # 模型是否可训练，默认值和`training`相同
    output_layer_num=4,  # 最后几层的输出将合并在一起作为最终的输出，只有当`training`是`False`有效
)
```

#### 关于`training`和`trainable`

虽然看起来相似，但这两个参数是不相关的。`training`表示是否在训练BERT语言模型，当为`True`时完整的BERT模型会被返回，当为`False`时没有MLM和NSP相关计算的结构，返回输入层和根据`output_layer_num`合并最后几层的输出。加载的层是否可训练只跟`trainable`有关。

### 使用Warmup

`AdamWarmup`优化器可用于学习率的「热身」与「衰减」。学习率将在`warmpup_steps`步线性增长到`lr`，并在总共`decay_steps`步后线性减少到`min_lr`。辅助函数`calc_train_steps`可用于计算这两个步数：

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

### 使用`tensorflow.python.keras`

在环境变量里加入`TF_KERAS=1`可以启用`tensorflow.python.keras`。

### 使用`theano`后端

在环境变量中加入`KERAS_BACKEND=theano`来启用`theano`后端。
