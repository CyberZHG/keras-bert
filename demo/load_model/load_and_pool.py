import os
import sys
import numpy as np
import keras
from keras_bert import load_vocabulary, load_trained_model_from_checkpoint, Tokenizer
from keras_bert.layers import MaskedGlobalMaxPool1D


if len(sys.argv) != 2:
    print('python load_model.py UNZIPPED_MODEL_PATH')
    sys.exit(-1)

print('This demo demonstrates how to load the pre-trained model and extract the sentence embedding with pooling.')

model_path = sys.argv[1]
config_path = os.path.join(model_path, 'bert_config.json')
checkpoint_path = os.path.join(model_path, 'bert_model.ckpt')
dict_path = os.path.join(model_path, 'vocab.txt')

model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=10)
pool_layer = MaskedGlobalMaxPool1D(name='Pooling')(model.output)
model = keras.models.Model(inputs=model.inputs, outputs=pool_layer)
model.summary(line_length=120)

token_dict = load_vocabulary(dict_path)

tokenizer = Tokenizer(token_dict)
text = '语言模型'
tokens = tokenizer.tokenize(text)
print('Tokens:', tokens)
indices, segments = tokenizer.encode(first=text, max_len=10)

predicts = model.predict([np.array([indices]), np.array([segments])])[0]
print('Pooled:', predicts.tolist()[:5])
