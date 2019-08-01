from keras_bert.backend import keras, initializers, regularizers, constraints
from keras_bert.backend import backend as K

__all__ = ['TaskEmbedding']


class TaskEmbedding(keras.layers.Layer):
    """Embedding for tasks.

        # Arguments
            input_dim: int > 0. Number of the tasks. Plus 1 if `mask_zero` is enabled.
            output_dim: int >= 0. Dimension of the dense embedding.
            embeddings_initializer: Initializer for the `embeddings` matrix.
            embeddings_regularizer: Regularizer function applied to the `embeddings` matrix.
            embeddings_constraint: Constraint function applied to the `embeddings` matrix.
            mask_zero: Generate zeros for 0 index if it is `True`.

        # Input shape
            Previous embeddings, 3D tensor with shape: `(batch_size, sequence_length, output_dim)`.
            Task IDs, 2D tensor with shape: `(batch_size, 1)`.

        # Output shape
            3D tensor with shape: `(batch_size, sequence_length, output_dim)`.
        """

    def __init__(self,
                 input_dim,
                 output_dim,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 embeddings_constraint=None,
                 mask_zero=False,
                 **kwargs):
        super(TaskEmbedding, self).__init__(**kwargs)
        self.supports_masking = True
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.embeddings_regularizer = regularizers.get(embeddings_regularizer)
        self.embeddings_constraint = constraints.get(embeddings_constraint)
        self.mask_zero = mask_zero

        self.embeddings = None

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer,
            regularizer=self.embeddings_regularizer,
            constraint=self.embeddings_constraint,
            name='embeddings',
        )
        super(TaskEmbedding, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        output_mask = None
        if mask is not None:
            output_mask = mask[0]
        return output_mask

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def call(self, inputs, **kwargs):
        inputs, tasks = inputs
        if K.dtype(tasks) != 'int32':
            tasks = K.cast(tasks, 'int32')
        task_embed = K.gather(self.embeddings, tasks)
        if self.mask_zero:
            task_embed = task_embed * K.expand_dims(K.cast(K.not_equal(tasks, 0), K.floatx()), axis=-1)
        if K.backend() == 'theano':
            task_embed = K.tile(task_embed, (1, K.shape(inputs)[1], 1))
        return inputs + task_embed

    def get_config(self):
        config = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'embeddings_initializer': initializers.serialize(self.embeddings_initializer),
            'embeddings_regularizer': regularizers.serialize(self.embeddings_regularizer),
            'embeddings_constraint': constraints.serialize(self.embeddings_constraint),
            'mask_zero': self.mask_zero,
        }
        base_config = super(TaskEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
