# Keras BERT

[![Version](https://img.shields.io/pypi/v/keras-bert.svg)](https://pypi.org/project/keras-bert/)
![License](https://img.shields.io/pypi/l/keras-bert.svg)

\[[中文](https://github.com/CyberZHG/keras-bert/blob/master/README.zh-CN.md)|[English](https://github.com/CyberZHG/keras-bert/blob/master/README.md)\]

[BERT](https://arxiv.org/pdf/1810.04805.pdf)的非官方实现，可以加载官方的预训练模型进行特征提取和预测。

## 安装

```bash
pip install keras-bert
```

## 使用

* [使用官方模型](#使用官方模型)
* [分词](#分词)
* [训练和使用](#训练和使用)
* [关于`training`和`trainable`](#关于training和trainable)
* [使用Warmup](#使用Warmup)
* [关于输入](#关于输入)
* [下载预训练模型](#下载预训练模型)
* [提取特征](#提取特征)
* [模型存储与加载](#模型存储与加载)
* [使用任务嵌入](#使用任务嵌入)
* [使用`tf.keras`](#使用tensorflowpythonkeras)

### External Links

* [Kashgari是一个极简且强大的 NLP 框架，可用于文本分类和标注的学习，研究及部署上线](https://github.com/BrikerMan/Kashgari)
* [当Bert遇上Keras：这可能是Bert最简单的打开姿势](https://spaces.ac.cn/archives/6736)
* [Keras ALBERT](https://github.com/TinkerMob/keras_albert_model)

### 使用官方模型

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
print(segments)  # 段落对应下标：`[0, 0, 0, 0, 0, 1, 1, 0, 0, 0]`
```

`Tokenizer`也提供了尝试去寻找分词后的结果在原始文本中的起始和终止下标的功能，输入可以是decode后的结果，包含少量的错词：

```python
from keras_bert import Tokenizer

intervals = Tokenizer.rematch("All rights reserved.", ["[UNK]", "righs", "[UNK]", "ser", "[UNK]", "[UNK]"])
# 结果是：[(0, 3), (4, 10), (11, 13), (13, 16), (16, 19), (19, 20)]
```

### 训练和使用

训练过程推荐使用官方的代码。这个代码库内包含一个的训练过程，`training`为`True`的情况下使用的是带warmup的Adam优化器：

```python
from tensorflow import keras
from keras_bert import get_base_dict, get_model, compile_model, gen_batch_inputs


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
compile_model(model)
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

此外，`trainable`可以是一个包含字符串的列表，如果某一层的前缀出现在列表中，则当前层是可训练的。在使用预训练模型时，如果不想再训练嵌入层，可以传入`trainable=['Encoder']`来只对编码层进行调整。

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

### 关于输入

在`training`为`True`的情况下，输入包含三项：token下标、segment下标、被masked的词的模版。当`training`为`False`时输入只包含前两项。位置下标由于是固定的，会在模型内部生成，不需要手动再输入一遍。被masked的词的模版在输入被masked的词是值为1，否则为0。

### 下载预训练模型

库中记录了一些预训练模型的下载地址，可以通过如下方式获得解压后的checkpoint的路径：

```python
from keras_bert import get_pretrained, PretrainedList, get_checkpoint_paths

model_path = get_pretrained(PretrainedList.multi_cased_base)
paths = get_checkpoint_paths(model_path)
print(paths.config, paths.checkpoint, paths.vocab)
```

### 提取特征

如果不需要微调，只想提取词/句子的特征，则可以使用`extract_embeddings`来简化流程。如提取每个句子对应的全部词的特征：

```python
from keras_bert import extract_embeddings

model_path = 'xxx/yyy/uncased_L-12_H-768_A-12'
texts = ['all work and no play', 'makes jack a dull boy~']

embeddings = extract_embeddings(model_path, texts)
```

返回的结果是一个list，长度和输入文本的个数相同，每个元素都是numpy的数组，默认会根据输出的长度进行裁剪，所以在这个例子中输出的大小分别为`(7, 768)`和`(8, 768)`。

如果输入是成对的句子，想使用最后4层特征，且提取`NSP`位输出和max-pooling的结果，则可以用：

```python
from keras_bert import extract_embeddings, POOL_NSP, POOL_MAX

model_path = 'xxx/yyy/uncased_L-12_H-768_A-12'
texts = [
    ('all work and no play', 'makes jack a dull boy'),
    ('makes jack a dull boy', 'all work and no play'),
]

embeddings = extract_embeddings(model_path, texts, output_layer_num=4, poolings=[POOL_NSP, POOL_MAX])
```

输出结果中不再包含词的特征，`NSP`和max-pooling的输出会拼接在一起，每个numpy数组的大小为`(768 x 4 x 2,)`。

第二个参数接受的是一个generator，如果想读取文件并生成特征，可以用下面的方法：

```python
import codecs
from keras_bert import extract_embeddings

model_path = 'xxx/yyy/uncased_L-12_H-768_A-12'

with codecs.open('xxx.txt', 'r', 'utf8') as reader:
    texts = map(lambda x: x.strip(), reader)
    embeddings = extract_embeddings(model_path, texts)
```

### 模型存储与加载

```python
from keras_bert import load_trained_model_from_checkpoint, get_custom_objects

model = load_trained_model_from_checkpoint('xxx', 'yyy')
model.save('save_path.h5')
model.load('save_path.h5', custom_objects=get_custom_objects())
```

### 使用任务嵌入

如果有多任务训练的需求，可以启用任务嵌入层，针对不同任务将嵌入的结果加上不同的编码，注意要让`Embedding-Task`层可训练：

```python
from keras_bert import get_pretrained, PretrainedList, get_checkpoint_paths, load_trained_model_from_checkpoint

model_path = get_pretrained(PretrainedList.multi_cased_base)
paths = get_checkpoint_paths(model_path)
model = load_trained_model_from_checkpoint(
    config_file=paths.config,
    checkpoint_file=paths.checkpoint,
    training=False,
    trainable=True,
    use_task_embed=True,
    task_num=10,
)
```
