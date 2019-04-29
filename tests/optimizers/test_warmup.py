from unittest import TestCase
import keras
import numpy as np
from keras_bert.optimizers import AdamWarmup


class TestWarmup(TestCase):

    def test_fit(self):
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
            optimizer=AdamWarmup(decay_steps=10000, warmup_steps=5000, lr=1e-3, min_lr=1e-4, amsgrad=True),
            loss='sparse_categorical_crossentropy',
        )
        model.fit(x, y, batch_size=10, epochs=110)
        results = model.predict(x).argmax(axis=-1)
        diff = np.sum(np.abs(y - results))
        self.assertLess(diff, 100)
