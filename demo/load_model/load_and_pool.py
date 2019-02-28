import sys
import codecs
import numpy as np
import keras
from keras_bert import load_trained_model_from_checkpoint
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

tokens = ['[CLS]', '语', '言', '模', '型', '[SEP]']

token_dict = {}
with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

token_input = np.asarray([[token_dict[token] for token in tokens] + [0] * (512 - len(tokens))])
seg_input = np.asarray([[0] * len(tokens) + [0] * (512 - len(tokens))])

print('Inputs:', token_input[0][:len(tokens)])

predicts = model.predict([token_input, seg_input])[0]
print('Pooled:', predicts.tolist()[:5])
