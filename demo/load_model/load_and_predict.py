import sys
import numpy as np
from keras_bert import load_vocabulary, load_trained_model_from_checkpoint, Tokenizer, get_checkpoint_paths

print('This demo demonstrates how to load the pre-trained model and check whether the two sentences are continuous')

if len(sys.argv) == 2:
    model_path = sys.argv[1]
else:
    from keras_bert.datasets import get_pretrained, PretrainedList
    model_path = get_pretrained(PretrainedList.chinese_base)

paths = get_checkpoint_paths(model_path)

model = load_trained_model_from_checkpoint(paths.config, paths.checkpoint, training=True, seq_len=None)
model.summary(line_length=120)

token_dict = load_vocabulary(paths.vocab)
token_dict_inv = {v: k for k, v in token_dict.items()}

tokenizer = Tokenizer(token_dict)
text = '数学是利用符号语言研究数量、结构、变化以及空间等概念的一门学科'
tokens = tokenizer.tokenize(text)
tokens[1] = tokens[2] = '[MASK]'
print('Tokens:', tokens)

indices = np.array([[token_dict[token] for token in tokens]])
segments = np.array([[0] * len(tokens)])
masks = np.array([[0, 1, 1] + [0] * (len(tokens) - 3)])

predicts = model.predict([indices, segments, masks])[0].argmax(axis=-1).tolist()
print('Fill with: ', list(map(lambda x: token_dict_inv[x], predicts[0][1:3])))


sentence_1 = '数学是利用符号语言研究數量、结构、变化以及空间等概念的一門学科。'
sentence_2 = '从某种角度看屬於形式科學的一種。'
print('Tokens:', tokenizer.tokenize(first=sentence_1, second=sentence_2))
indices, segments = tokenizer.encode(first=sentence_1, second=sentence_2)
masks = np.array([[0] * len(indices)])

predicts = model.predict([np.array([indices]), np.array([segments]), masks])[1]
print('%s is random next: ' % sentence_2, bool(np.argmax(predicts, axis=-1)[0]))

sentence_2 = '任何一个希尔伯特空间都有一族标准正交基。'
print('Tokens:', tokenizer.tokenize(first=sentence_1, second=sentence_2))
indices, segments = tokenizer.encode(first=sentence_1, second=sentence_2)
masks = np.array([[0] * len(indices)])

predicts = model.predict([np.array([indices]), np.array([segments]), masks])[1]
print('%s is random next: ' % sentence_2, bool(np.argmax(predicts, axis=-1)[0]))
