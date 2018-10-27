import unittest
import keras
import numpy as np
from keras_bert.layers import Transformer


class TestTransfomer(unittest.TestCase):

    def test_sample(self):
        input_layer = keras.layers.Input(
            shape=(512,),
            name='Input',
        )
        embed_layer = keras.layers.Embedding(
            input_dim=12,
            output_dim=768,
            mask_zero=True,
            name='Embedding',
        )(input_layer)
        output_layer = Transformer(
            head_num=12,
            hidden_dim=768 * 4,
            dropout_rate=0.001,
            name='Transformer',
        )(embed_layer)
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mse'],
        )
        model.summary(line_length=120)
        self.assertEqual((None, 512, 768), model.layers[-1].output_shape)

    def test_fit(self):
        input_layer = keras.layers.Input(
            shape=(2, 3),
            name='Input',
        )
        dense_layer = keras.layers.Dense(units=3, name='Dense-1')(input_layer)
        transformer_layer = Transformer(
            head_num=3,
            hidden_dim=12,
            dropout_rate=0.001,
            name='Transformer-1',
        )(dense_layer)
        transformer_layer = Transformer(
            head_num=3,
            hidden_dim=12,
            dropout_rate=0.001,
            name='Transformer-2',
        )(transformer_layer)
        dense_layer = keras.layers.Dense(units=3, name='Dense-2')(transformer_layer)
        model = keras.models.Model(
            inputs=input_layer,
            outputs=dense_layer,
        )
        model.compile(
            optimizer=keras.optimizers.Adam(lr=1e-3),
            loss='mse',
            metrics=['mse'],
        )
        model.summary()

        def _generator(batch_size=32):
            while True:
                inputs = np.random.random((batch_size, 2, 3))
                outputs = np.asarray([[[0.0, -0.1, 0.2]] * 2] * batch_size)
                yield inputs, outputs

        model.fit_generator(
            generator=_generator(),
            steps_per_epoch=1000,
            epochs=10,
            validation_data=_generator(),
            validation_steps=100,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
            ],
        )
        for inputs, _ in _generator(batch_size=3):
            predicts = model.predict(inputs)
            expect = np.round(np.asarray([[[0.0, -0.1, 0.2]] * 2] * 3), decimals=1)
            actual = np.round(predicts, decimals=1)
            self.assertTrue(np.allclose(expect, actual), (expect, actual))
            break
