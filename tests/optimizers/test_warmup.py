import os
import tempfile
from unittest import TestCase
import numpy as np
from keras_bert.backend import keras, TF_KERAS
from keras_bert import AdamWarmup


class TestWarmup(TestCase):

    def _test_fit(self, optmizer):
        x = np.random.standard_normal((1000, 5))
        y = np.dot(x, np.random.standard_normal((5, 2))).argmax(axis=-1)
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(
            units=2,
            input_shape=(5,),
            kernel_constraint=keras.constraints.MaxNorm(1000.0),
            activation='softmax',
        ))
        model.compile(
            optimizer=optmizer,
            loss='sparse_categorical_crossentropy',
        )
        model.fit(
            x, y,
            batch_size=10,
            epochs=110,
            callbacks=[keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-4, patience=3)],
        )

        model_path = os.path.join(tempfile.gettempdir(), 'keras_warmup_%f.h5' % np.random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects={'AdamWarmup': AdamWarmup})

        results = model.predict(x).argmax(axis=-1)
        diff = np.sum(np.abs(y - results))
        self.assertLess(diff, 100)

    def test_fit(self):
        self._test_fit(AdamWarmup(
            decay_steps=10000,
            warmup_steps=5000,
            learning_rate=1e-3,
            min_lr=1e-4,
            amsgrad=False,
            weight_decay=1e-3,
        ))

    def test_fit_amsgrad(self):
        self._test_fit(AdamWarmup(
            decay_steps=10000,
            warmup_steps=5000,
            learning_rate=1e-3,
            min_lr=1e-4,
            amsgrad=True,
            weight_decay=1e-3,
        ))

    def test_fit_embed(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Embedding(
            input_shape=(None,),
            input_dim=5,
            output_dim=16,
            mask_zero=True,
        ))
        model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=8)))
        model.add(keras.layers.Dense(units=2, activation='softmax'))
        model.compile(AdamWarmup(
            decay_steps=10000,
            warmup_steps=5000,
            learning_rate=1e-3,
            min_lr=1e-4,
            amsgrad=True,
            weight_decay=1e-3,
        ), loss='sparse_categorical_crossentropy')

        x = np.random.randint(0, 5, (1024, 15))
        y = (x[:, 1] > 2).astype('int32')
        model.fit(x, y, epochs=10)

        model_path = os.path.join(tempfile.gettempdir(), 'test_warmup_%f.h5' % np.random.random())
        model.save(model_path)
        keras.models.load_model(model_path, custom_objects={'AdamWarmup': AdamWarmup})

    def test_legacy(self):
        opt = AdamWarmup(
            decay_steps=10000,
            warmup_steps=5000,
            learning_rate=1e-3,
        )
        if not TF_KERAS:
            opt.lr = opt.lr
