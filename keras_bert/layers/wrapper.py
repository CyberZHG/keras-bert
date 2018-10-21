import keras


class Wrapper(keras.layers.Layer):

    CONFIG_PREFIX = 'wrapper_layer_'

    def __init__(self, layers=None, **kwargs):
        if layers is None:
            self.layers = {}
        else:
            self.layers = layers
            self.built = True
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

    def get_config(self):
        config = {}
        for name, layer in self.layers.items():
            config[self.CONFIG_PREFIX + name] = {
                'class_name': layer.__class__.__name__,
                'config': layer.get_config(),
            }
        base_config = super(Wrapper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        layers = {}
        keys = list(filter(lambda key: key.startswith(Wrapper.CONFIG_PREFIX), config.keys()))
        for key in keys:
            if key.startswith(Wrapper.CONFIG_PREFIX):
                name = key[len(Wrapper.CONFIG_PREFIX):]
                layers[name] = keras.layers.deserialize(config.pop(key), custom_objects=custom_objects)
        return cls(layers=layers, **config)
