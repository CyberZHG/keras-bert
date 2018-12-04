import json
import keras
import numpy as np
import tensorflow as tf
from .bert import get_model


def load_trained_model_from_checkpoint(config_file, checkpoint_file, training=False):
    """Load trained official model from checkpoint.

    :param config_file: The path to the JSON configuration file.
    :param checkpoint_file: The path to the checkpoint files, should end with '.ckpt'.
    :param training: If training, the whole model will be returned.
                     Otherwise, the MLM and NSP parts will be ignored.
    :return:
    """
    with open(config_file, 'r') as reader:
        config = json.loads(reader.read())
    model = get_model(
        token_num=config['vocab_size'],
        pos_num=config['max_position_embeddings'],
        seq_len=config['max_position_embeddings'],
        embed_dim=config['hidden_size'],
        transformer_num=config['num_hidden_layers'],
        head_num=config['num_attention_heads'],
        feed_forward_dim=config['intermediate_size'],
        training=training,
    )
    if not training:
        inputs, outputs = model
        model = keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics={})
    model.get_layer(name='Embedding-Token').set_weights([
        tf.train.load_variable(checkpoint_file, 'bert/embeddings/word_embeddings'),
    ])
    model.get_layer(name='Embedding-Position').set_weights([
        tf.train.load_variable(checkpoint_file, 'bert/embeddings/position_embeddings'),
    ])
    model.get_layer(name='Embedding-Segment').set_weights([
        tf.train.load_variable(checkpoint_file, 'bert/embeddings/token_type_embeddings'),
    ])
    model.get_layer(name='Embedding-Norm').set_weights([
        tf.train.load_variable(checkpoint_file, 'bert/embeddings/LayerNorm/gamma'),
        tf.train.load_variable(checkpoint_file, 'bert/embeddings/LayerNorm/beta'),
    ])
    for i in range(config['num_hidden_layers']):
        model.get_layer(name='Encoder-%d-MultiHeadSelfAttention' % (i + 1)).set_weights([
            tf.train.load_variable(checkpoint_file, 'bert/encoder/layer_%d/attention/self/query/kernel' % i),
            tf.train.load_variable(checkpoint_file, 'bert/encoder/layer_%d/attention/self/query/bias' % i),
            tf.train.load_variable(checkpoint_file, 'bert/encoder/layer_%d/attention/self/key/kernel' % i),
            tf.train.load_variable(checkpoint_file, 'bert/encoder/layer_%d/attention/self/key/bias' % i),
            tf.train.load_variable(checkpoint_file, 'bert/encoder/layer_%d/attention/self/value/kernel' % i),
            tf.train.load_variable(checkpoint_file, 'bert/encoder/layer_%d/attention/self/value/bias' % i),
            tf.train.load_variable(checkpoint_file, 'bert/encoder/layer_%d/attention/output/dense/kernel' % i),
            tf.train.load_variable(checkpoint_file, 'bert/encoder/layer_%d/attention/output/dense/bias' % i),
        ])
        model.get_layer(name='Encoder-%d-MultiHeadSelfAttention-Norm' % (i + 1)).set_weights([
            tf.train.load_variable(checkpoint_file, 'bert/encoder/layer_%d/attention/output/LayerNorm/gamma' % i),
            tf.train.load_variable(checkpoint_file, 'bert/encoder/layer_%d/attention/output/LayerNorm/beta' % i),
        ])
        model.get_layer(name='Encoder-%d-MultiHeadSelfAttention-Norm' % (i + 1)).set_weights([
            tf.train.load_variable(checkpoint_file, 'bert/encoder/layer_%d/attention/output/LayerNorm/gamma' % i),
            tf.train.load_variable(checkpoint_file, 'bert/encoder/layer_%d/attention/output/LayerNorm/beta' % i),
        ])
        model.get_layer(name='Encoder-%d-FeedForward' % (i + 1)).set_weights([
            tf.train.load_variable(checkpoint_file, 'bert/encoder/layer_%d/intermediate/dense/kernel' % i),
            tf.train.load_variable(checkpoint_file, 'bert/encoder/layer_%d/intermediate/dense/bias' % i),
            tf.train.load_variable(checkpoint_file, 'bert/encoder/layer_%d/output/dense/kernel' % i),
            tf.train.load_variable(checkpoint_file, 'bert/encoder/layer_%d/output/dense/bias' % i),
        ])
        model.get_layer(name='Encoder-%d-FeedForward-Norm' % (i + 1)).set_weights([
            tf.train.load_variable(checkpoint_file, 'bert/encoder/layer_%d/output/LayerNorm/gamma' % i),
            tf.train.load_variable(checkpoint_file, 'bert/encoder/layer_%d/output/LayerNorm/beta' % i),
        ])
    if training:
        model.get_layer(name='MLM-Dense').set_weights([
            tf.train.load_variable(checkpoint_file, 'cls/predictions/transform/dense/kernel'),
            tf.train.load_variable(checkpoint_file, 'cls/predictions/transform/dense/bias'),
        ])
        model.get_layer(name='MLM-Norm').set_weights([
            tf.train.load_variable(checkpoint_file, 'cls/predictions/transform/LayerNorm/gamma'),
            tf.train.load_variable(checkpoint_file, 'cls/predictions/transform/LayerNorm/beta'),
        ])
        model.get_layer(name='MLM-Sim').set_weights([
            tf.train.load_variable(checkpoint_file, 'cls/predictions/output_bias'),
        ])
        model.get_layer(name='NSP-Dense').set_weights([
            tf.train.load_variable(checkpoint_file, 'bert/pooler/dense/kernel'),
            tf.train.load_variable(checkpoint_file, 'bert/pooler/dense/bias'),
        ])
        model.get_layer(name='NSP').set_weights([
            np.transpose(tf.train.load_variable(checkpoint_file, 'cls/seq_relationship/output_weights')),
            tf.train.load_variable(checkpoint_file, 'cls/seq_relationship/output_bias'),
        ])
    return model
