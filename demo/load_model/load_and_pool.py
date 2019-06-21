import sys
import numpy as np
import keras
from keras_bert import load_vocabulary, load_trained_model_from_checkpoint, Tokenizer, get_checkpoint_paths
from keras_bert.layers import MaskedGlobalMaxPool1D

print('This demo demonstrates how to load the pre-trained model and extract the sentence embedding with pooling.')

if len(sys.argv) == 2:
    model_path = sys.argv[1]
else:
    from keras_bert.datasets import get_pretrained, PretrainedList
    model_path = get_pretrained(PretrainedList.chinese_base)

paths = get_checkpoint_paths(model_path)

model = load_trained_model_from_checkpoint(paths.config, paths.checkpoint, seq_len=10)
pool_layer = MaskedGlobalMaxPool1D(name='Pooling')(model.output)
model = keras.models.Model(inputs=model.inputs, outputs=pool_layer)
model.summary(line_length=120)

token_dict = load_vocabulary(paths.vocab)

tokenizer = Tokenizer(token_dict)
text = '语言模型'
tokens = tokenizer.tokenize(text)
print('Tokens:', tokens)
indices, segments = tokenizer.encode(first=text, max_len=10)

predicts = model.predict([np.array([indices]), np.array([segments])])[0]
print('Pooled:', predicts.tolist()[:5])
