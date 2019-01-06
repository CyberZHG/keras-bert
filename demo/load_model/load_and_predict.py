import sys
import codecs
import numpy as np
from keras_bert import load_trained_model_from_checkpoint


if len(sys.argv) != 4:
    print('python load_model.py CONFIG_PATH CHECKPOINT_PATH DICT_PATH')
    print('CONFIG_PATH:     $UNZIPPED_MODEL_PATH/bert_config.json')
    print('CHECKPOINT_PATH: $UNZIPPED_MODEL_PATH/bert_model.ckpt')
    print('DICT_PATH:       $UNZIPPED_MODEL_PATH/vocab.txt')
    sys.exit(-1)

config_path, checkpoint_path, dict_path = tuple(sys.argv[1:])

model = load_trained_model_from_checkpoint(config_path, checkpoint_path, training=True)
model.summary(line_length=120)

tokens = ['[CLS]', '[MASK]', '[MASK]'] + list('是利用符号语言研究数量、结构、变化以及空间等概念的一门学科') + ['[SEP]']

token_dict = {}
with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)
token_dict_rev = {v: k for k, v in token_dict.items()}

token_input = np.asarray([[token_dict[token] for token in tokens] + [0] * (512 - len(tokens))])
seg_input = np.asarray([[0] * len(tokens) + [0] * (512 - len(tokens))])
mask_input = np.asarray([[0, 1, 1] + [0] * (512 - 3)])

print(token_input[0][:len(tokens)])

predicts = model.predict([token_input, seg_input, mask_input])[0]
predicts = np.argmax(predicts, axis=-1)
print(predicts[0][:len(tokens)])
print(list(map(lambda x: token_dict_rev[x], predicts[0][1:3])))


sentence_1 = '数学是利用符号语言研究數量、结构、变化以及空间等概念的一門学科。'
sentence_2 = '从某种角度看屬於形式科學的一種。'

tokens = ['[CLS]'] + list(sentence_1) + ['[SEP]'] + list(sentence_2) + ['[SEP]']

token_input = np.asarray([[token_dict[token] for token in tokens] + [0] * (512 - len(tokens))])
seg_input = np.asarray([[0] * (len(sentence_1) + 2) + [1] * (len(sentence_2) + 1) + [0] * (512 - len(tokens))])
mask_input = np.asarray([[0] * 512])

predicts = model.predict([token_input, seg_input, mask_input])[1]
print('%s is random next: ' % sentence_2, bool(np.argmax(predicts, axis=-1)[0]))

sentence_2 = '任何一个希尔伯特空间都有一族标准正交基。'

tokens = ['[CLS]'] + list(sentence_1) + ['[SEP]'] + list(sentence_2) + ['[SEP]']

token_input = np.asarray([[token_dict[token] for token in tokens] + [0] * (512 - len(tokens))])
seg_input = np.asarray([[0] * (len(sentence_1) + 2) + [1] * (len(sentence_2) + 1) + [0] * (512 - len(tokens))])
mask_input = np.asarray([[0] * 512])

predicts = model.predict([token_input, seg_input, mask_input])[1]
print('%s is random next: ' % sentence_2, bool(np.argmax(predicts, axis=-1)[0]))
