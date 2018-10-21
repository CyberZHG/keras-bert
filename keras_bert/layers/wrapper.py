import keras


class Wrapper(keras.layers.Layer):

    def __init__(self, **kwargs):
        self.layers = {}
        super(Wrapper, self).__init__(**kwargs)

    @property
    def trainable_weights(self):
        weights = []
        for key in sorted(self.layers.keys()):
            weights += self.layers[key].trainable_weights
        return weights

    @property
    def non_trainable_weights(self):
        weights = []
        for key in sorted(self.layers.keys()):
            weights += self.layers[key].non_trainable_weights
        return weights
