import unittest
import random
import keras
import numpy as np
from keras_bert.layers import get_inputs, Embeddings, Transformer, Masked


class TestMasked(unittest.TestCase):

    def test_sample(self):
        inputs = get_inputs(seq_len=512)
        embed_layer = Embeddings(input_dim=12, output_dim=768, position_dim=512)(inputs)
        masked_layer = Masked(name='Masked')([embed_layer, inputs[-1]])
        model = keras.models.Model(inputs=inputs, outputs=masked_layer)
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mse'],
        )
        model.summary(line_length=120)
        model.predict([
            np.asarray([[1] + [0] * 511]),
            np.asarray([[0] * 512]),
            np.asarray([[0] * 512]),
            np.asarray([[1] + [0] * 511]),
        ])
        self.assertEqual((None, 512, 768), model.layers[-1].output_shape)

    def test_mask_result(self):
        input_layer = keras.layers.Input(
            shape=(None,),
            name='Input',
        )
        embed_layer = keras.layers.Embedding(
            input_dim=12,
            output_dim=9,
            mask_zero=True,
            name='Embedding',
        )(input_layer)
        transformer_layer = Transformer(
            head_num=1,
            hidden_dim=12,
            dropout_rate=0.1,
            name='Transformer',
        )(embed_layer)
        dense_layer = keras.layers.Dense(
            units=12,
            activation='softmax',
            name='Dense',
        )(transformer_layer)
        mask_layer = keras.layers.Input(
            shape=(None,),
            name='Mask',
        )
        masked_layer, mask_result = Masked(
            return_masked=True,
            name='Masked',
        )([dense_layer, mask_layer])
        model = keras.models.Model(
            inputs=[input_layer, mask_layer],
            outputs=[masked_layer, mask_result],
        )
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mse'],
        )
        model.summary(line_length=150)
        predicts = model.predict([
            np.asarray([
                [1, 2, 3, 4, 5, 6, 7, 8, 0, 0],
                [1, 2, 3, 4, 0, 0, 0, 0, 0, 0],
            ]),
            np.asarray([
                [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
            ]),
        ])
        expect = np.asarray([
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        ])
        self.assertTrue(np.allclose(expect, predicts[1]))

    def test_fit(self):
        input_layer = keras.layers.Input(
            shape=(15,),
            name='Input',
        )
        embed_layer = keras.layers.Embedding(
            input_dim=12,
            output_dim=24,
            mask_zero=True,
            name='Embedding',
        )(input_layer)
        rnn_layer = keras.layers.Bidirectional(
            keras.layers.LSTM(units=100, return_sequences=True),
            name='Bi-LSTM',
        )(embed_layer)
        dense_layer = keras.layers.Dense(
            units=12,
            activation='softmax',
            name='Dense',
        )(rnn_layer)
        mask_layer = keras.layers.Input(
            shape=(None,),
            name='Mask',
        )
        masked_layer = Masked(
            name='Masked',
        )([dense_layer, mask_layer])
        model = keras.models.Model(
            inputs=[input_layer, mask_layer],
            outputs=masked_layer,
        )
        model.compile(
            optimizer=keras.optimizers.Adam(lr=1e-4),
            loss=keras.losses.sparse_categorical_crossentropy,
            metrics=[keras.metrics.sparse_categorical_crossentropy],
        )
        model.summary(line_length=150)

        def _generator(batch_size=32):
            while True:
                inputs, masked, outputs = [], [], []
                for _ in range(batch_size):
                    inputs.append([])
                    masked.append([])
                    outputs.append([])
                    has_mask = False
                    for i in range(1, 11):
                        inputs[-1].append(i)
                        outputs[-1].append([i])
                        if random.random() < 0.3:
                            has_mask = True
                            inputs[-1][-1] = 11
                            masked[-1].append(1)
                        else:
                            masked[-1].append(0)
                    if not has_mask:
                        masked[-1][0] = 1
                    inputs[-1] += [0] * (15 - len(inputs[-1]))
                    masked[-1] += [0] * (15 - len(masked[-1]))
                    outputs[-1] += [[0]] * (15 - len(outputs[-1]))
                yield [np.asarray(inputs), np.asarray(masked)], np.asarray(outputs)

        model.fit_generator(
            generator=_generator(),
            steps_per_epoch=1000,
            epochs=10,
            validation_data=_generator(),
            validation_steps=100,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
            ],
        )
        for inputs, outputs in _generator(batch_size=32):
            predicts = model.predict(inputs)
            actual = np.argmax(predicts, axis=-1)
            for i in range(32):
                for j in range(15):
                    if inputs[1][i][j]:
                        self.assertEqual(j + 1, actual[i][j])
            break
