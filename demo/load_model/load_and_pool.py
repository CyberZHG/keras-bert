import sys
import codecs
import numpy as np
import keras
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras_bert.layers import MaskedGlobalMaxPool1D


if len(sys.argv) != 4:
    print('python load_model.py CONFIG_PATH CHECKPOINT_PATH DICT_PATH')
    print('CONFIG_PATH:     $UNZIPPED_MODEL_PATH/bert_config.json')
    print('CHECKPOINT_PATH: $UNZIPPED_MODEL_PATH/bert_model.ckpt')
    print('DICT_PATH:       $UNZIPPED_MODEL_PATH/vocab.txt')
    sys.exit(-1)

print('This demo demonstrates how to load the pre-trained model and extract the sentence embedding with pooling.')

config_path, checkpoint_path, dict_path = tuple(sys.argv[1:])

model = load_trained_model_from_checkpoint(config_path, checkpoint_path)
pool_layer = MaskedGlobalMaxPool1D(name='Pooling')(model.output)
model = keras.models.Model(inputs=model.inputs, outputs=pool_layer)
model.summary(line_length=120)

token_dict = {}
with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

tokenizer = Tokenizer(token_dict)
text = '语言模型'
tokens = tokenizer.tokenize(text)
print('Tokens:', tokens)
indices, segments = tokenizer.encode(first='语言模型', max_len=512)

predicts = model.predict([np.array([indices]), np.array([segments])])[0]
print('Pooled:', predicts.tolist()[:5])
