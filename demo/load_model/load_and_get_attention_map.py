import sys
import numpy as np
from keras_bert import load_vocabulary, load_trained_model_from_checkpoint, Tokenizer, get_checkpoint_paths
from keras_bert.backend import backend as K

print('This demo demonstrates how to load the pre-trained model and extract the attention map')

if len(sys.argv) == 2:
    model_path = sys.argv[1]
else:
    from keras_bert.datasets import get_pretrained, PretrainedList
    model_path = get_pretrained(PretrainedList.chinese_base)

paths = get_checkpoint_paths(model_path)

model = load_trained_model_from_checkpoint(paths.config, paths.checkpoint, seq_len=10)
attention_layer = model.get_layer('Encoder-1-MultiHeadSelfAttention')
model = K.function(model.inputs, attention_layer.attention)

token_dict = load_vocabulary(paths.vocab)

tokenizer = Tokenizer(token_dict)
text = '语言模型'
tokens = tokenizer.tokenize(text)
print('Tokens:', tokens)
indices, segments = tokenizer.encode(first=text, max_len=10)

predicts = model([np.array([indices]), np.array([segments])])[0]
for i, token in enumerate(tokens):
    print(token)
    for head_index in range(12):
        print(predicts[i][head_index, :len(text) + 2].tolist())
