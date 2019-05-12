import sys
import os
import codecs
import numpy as np
from keras_bert import load_trained_model_from_checkpoint, Tokenizer


if len(sys.argv) != 4:
    print('python load_model.py CONFIG_PATH CHECKPOINT_PATH DICT_PATH')
    print('CONFIG_PATH:     $UNZIPPED_MODEL_PATH/bert_config.json')
    print('CHECKPOINT_PATH: $UNZIPPED_MODEL_PATH/bert_model.ckpt')
    print('DICT_PATH:       $UNZIPPED_MODEL_PATH/vocab.txt')
    sys.argv = [
       sys.argv[0],
       os.environ.get('CONFIG_PATH', None) or os.path.join(os.environ['UNZIPPED_MODEL_PATH'], 'bert_config.json'),
       os.environ.get('CHECKPOINT_PATH', None) or os.path.join(os.environ['UNZIPPED_MODEL_PATH'], 'bert_model.ckpt'),
       os.environ.get('DICT_PATH', None) or os.path.join(os.environ['UNZIPPED_MODEL_PATH'], 'vocab.txt'),
    ]

print('This demo demonstrates how to load the pre-trained model and check whether the two sentences are continuous')

config_path, checkpoint_path, dict_path = tuple(sys.argv[1:])

model = load_trained_model_from_checkpoint(config_path, checkpoint_path, training=True)
model.summary(line_length=120)

token_dict = {}
with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)
token_dict_rev = {v: k for k, v in token_dict.items()}

tokenizer = Tokenizer(token_dict)
text = '数学是利用符号语言研究數量、结构、变化以及空间等概念的一門学科。'
text = 'Google is dedicated to proactive openness and applying machine learning technology to further the common good.'
# original keras-bert demo example in chinese translated to English:
text = 'Mathematics is a discipline that uses symbolic language to study concepts such as quantity, structure, change, and space.'
tokens = tokenizer.tokenize(text)
tokens[1] = tokens[2] = '[MASK]'
print('Tokens:', tokens)

indices = np.asarray([[token_dict[token] for token in tokens] + [0] * (512 - len(tokens))])
segments = np.asarray([[0] * len(tokens) + [0] * (512 - len(tokens))])
masks = np.asarray([[0, 1, 1] + [0] * (512 - 3)])

predicts = model.predict([indices, segments, masks])[0]
predicts = np.argmax(predicts, axis=-1)
print('Fill with: ', list(map(lambda x: token_dict_rev[x], predicts[0][1:3])))


sentence_1 = '数学是利用符号语言研究數量、结构、变化以及空间等概念的一門学科。'
sentence_2 = '从某种角度看屬於形式科學的一種。'
sentence_1 = text
sentence_2 = 'Joseph Conrad said "We live as we dream, alone." '
print('Tokens:', tokenizer.tokenize(first=sentence_1, second=sentence_2))
indices, segments = tokenizer.encode(first=sentence_1, second=sentence_2, max_len=512)
masks = np.array([[0] * 512])

predicts = model.predict([np.array([indices]), np.array([segments]), masks])[1]
print('%s is random next: ' % sentence_2, bool(np.argmax(predicts, axis=-1)[0]))

sentence_2 = '任何一个希尔伯特空间都有一族标准正交基。'
sentence_2 = 'A neural network is a computational graph trained on the statistics of a dataset to form a mathematical model. '
sentence_2 = 'Mathematicians seek and use patterns to formulate new conjectures; they resolve the truth or falsity of conjectures by mathematical proof. '


print('Tokens:', tokenizer.tokenize(first=sentence_1, second=sentence_2))
indices, segments = tokenizer.encode(first=sentence_1, second=sentence_2, max_len=512)

predicts = model.predict([np.array([indices]), np.array([segments]), masks])[1]
print('%s is random next: ' % sentence_2, bool(np.argmax(predicts, axis=-1)[0]))
