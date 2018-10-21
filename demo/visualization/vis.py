import keras
from keras_bert import get_model


model = get_model(
    token_num=300,
    transformer_num=3,
    head_num=4,
)
keras.utils.plot_model(model, show_shapes=True, to_file='bert.png')
