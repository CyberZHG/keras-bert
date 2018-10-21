import os
import keras
from keras_bert import get_model


model = get_model(
    token_num=30000,
    pos_num=512,
    transformer_num=12,
    head_num=12,
    embed_dim=768,
    feed_forward_dim=768 * 4,
)
model.summary(line_length=120)
current_path = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(current_path, 'bert_small.png')
keras.utils.plot_model(model, show_shapes=True, to_file=output_path)

model = get_model(
    token_num=30000,
    pos_num=512,
    transformer_num=24,
    head_num=16,
    embed_dim=1024,
    feed_forward_dim=1024 * 4,
)
model.summary(line_length=120)
output_path = os.path.join(current_path, 'bert_big.png')
keras.utils.plot_model(model, show_shapes=True, to_file=output_path)
