import keras
from keras_bert.layers import (get_inputs, get_embedding, get_transformer,
                               Attention, FeedForward, Masked, Extract, LayerNormalization)


def get_model(token_num,
              pos_num=512,
              seq_len=512,
              embed_dim=768,
              transformer_num=12,
              head_num=12,
              feed_forward_dim=3072,
              dropout=0.1):
    inputs = get_inputs(seq_len=seq_len)
    embed_layer = get_embedding(
        inputs=inputs,
        token_num=token_num,
        pos_num=pos_num,
        embed_dim=embed_dim,
        dropout=dropout,
    )
    transformed = embed_layer
    for i in range(transformer_num):
        transformed = get_transformer(
            inputs=transformed,
            head_num=head_num,
            hidden_dim=feed_forward_dim,
            name='Transformer-%d' % (i + 1),
            dropout=dropout,
        )
    mlm_pred_layer = keras.layers.Dense(
        units=token_num,
        activation='softmax',
        name='Dense-MLM',
    )(transformed)
    masked_layer = Masked(name='MLM')([mlm_pred_layer, inputs[-1]])
    extract_layer = Extract(index=0, name='Extract')(transformed)
    nsp_pred_layer = keras.layers.Dense(
        units=2,
        activation='softmax',
        name='NSP',
    )(extract_layer)
    model = keras.models.Model(inputs=inputs, outputs=[masked_layer, nsp_pred_layer])
    model.compile(
        optimizer=keras.optimizers.Adam(lr=1e-4),
        loss=keras.losses.sparse_categorical_crossentropy,
        metrics=[keras.losses.sparse_categorical_crossentropy],
    )
    return model


def get_custom_objects():
    return {
        'Attention': Attention,
        'FeedForward': FeedForward,
        'LayerNormalization': LayerNormalization,
        'Masked': Masked,
        'Extract': Extract,
    }
