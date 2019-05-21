import sys
import codecs
import numpy as np
from keras_bert import load_trained_model_from_checkpoint, Tokenizer


if len(sys.argv) != 4:
    print('python load_model.py CONFIG_PATH CHECKPOINT_PATH DICT_PATH')
    print('CONFIG_PATH:     $UNZIPPED_MODEL_PATH/bert_config.json')
    print('CHECKPOINT_PATH: $UNZIPPED_MODEL_PATH/bert_model.ckpt')
    print('DICT_PATH:       $UNZIPPED_MODEL_PATH/vocab.txt')
    sys.exit(-1)

print('This demo demonstrates how to load the pre-trained model and check whether the two sentences are continuous')

config_path, checkpoint_path, dict_path = tuple(sys.argv[1:])

model = load_trained_model_from_checkpoint(config_path, checkpoint_path, training=True)
model.summary(line_length=120)

token_dict = {}
with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)
token_dict_inv = {v: k for k, v in token_dict.items()}

tokenizer = Tokenizer(token_dict)
text = '数学是利用符号语言研究数量、结构、变化以及空间等概念的一门学科'
tokens = tokenizer.tokenize(text)
tokens[1] = tokens[2] = '[MASK]'
print('Tokens:', tokens)

indices = np.array([[token_dict[token] for token in tokens] + [0] * (512 - len(tokens))])
segments = np.array([[0] * len(tokens) + [0] * (512 - len(tokens))])
masks = np.array([[0, 1, 1] + [0] * (512 - 3)])

predicts = model.predict([indices, segments, masks])[0].argmax(axis=-1).tolist()
print('Fill with: ', list(map(lambda x: token_dict_inv[x], predicts[0][1:3])))


sentence_1 = '数学是利用符号语言研究數量、结构、变化以及空间等概念的一門学科。'
sentence_2 = '从某种角度看屬於形式科學的一種。'
print('Tokens:', tokenizer.tokenize(first=sentence_1, second=sentence_2))
indices, segments = tokenizer.encode(first=sentence_1, second=sentence_2, max_len=512)
masks = np.array([[0] * 512])

predicts = model.predict([np.array([indices]), np.array([segments]), masks])[1]
print('%s is random next: ' % sentence_2, bool(np.argmax(predicts, axis=-1)[0]))

sentence_2 = '任何一个希尔伯特空间都有一族标准正交基。'
print('Tokens:', tokenizer.tokenize(first=sentence_1, second=sentence_2))
indices, segments = tokenizer.encode(first=sentence_1, second=sentence_2, max_len=512)

predicts = model.predict([np.array([indices]), np.array([segments]), masks])[1]
print('%s is random next: ' % sentence_2, bool(np.argmax(predicts, axis=-1)[0]))
