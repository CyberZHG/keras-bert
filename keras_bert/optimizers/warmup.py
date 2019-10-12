from keras_bert.backend import keras
from keras_bert.backend import backend as K

__all__ = ['AdamWarmup']


def identity(x):
    return x


symbolic = identity
if hasattr(K, 'symbolic'):
    symbolic = K.symbolic


class AdamWarmup(keras.optimizers.Optimizer):
    """Adam optimizer with warmup.

    Default parameters follow those provided in the original paper.

    # Arguments
        decay_steps: Learning rate will decay linearly to zero in decay steps.
        warmup_steps: Learning rate will increase linearly to lr in first warmup steps.
        learning_rate: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        weight_decay: float >= 0. Weight decay.
        weight_decay_pattern: A list of strings. The substring of weight names to be decayed.
                              All weights will be decayed if it is None.
        amsgrad: boolean. Whether to apply the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and
            Beyond".
    """

    def __init__(self, decay_steps, warmup_steps, min_lr=0.0,
                 learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, weight_decay=0., weight_decay_pattern=None,
                 amsgrad=False, **kwargs):
        learning_rate = kwargs.pop('lr', learning_rate)
        super(AdamWarmup, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.decay_steps = K.variable(decay_steps, name='decay_steps')
            self.warmup_steps = K.variable(warmup_steps, name='warmup_steps')
            self.min_lr = K.variable(min_lr, name='min_lr')
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.learning_rate = K.variable(learning_rate, name='learning_rate')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.weight_decay = K.variable(weight_decay, name='weight_decay')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_weight_decay = weight_decay
        self.weight_decay_pattern = weight_decay_pattern
        self.amsgrad = amsgrad

    @property
    def lr(self):
        return self.learning_rate

    @lr.setter
    def lr(self, learning_rate):
        self.learning_rate = learning_rate

    @symbolic
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        t = K.cast(self.iterations, K.floatx()) + 1

        lr = K.switch(
            t <= self.warmup_steps,
            self.lr * (t / self.warmup_steps),
            self.min_lr + (self.lr - self.min_lr) * (1.0 - K.minimum(t, self.decay_steps) / self.decay_steps),
        )

        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p), name='m_{}'.format(i)) for i, p in enumerate(params)]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p), name='v_{}'.format(i)) for i, p in enumerate(params)]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p), name='vh_{}'.format(i)) for i, p in enumerate(params)]
        else:
            vhats = [K.zeros(1, dtype=K.dtype(p), name='vh_{}'.format(i)) for i, p in enumerate(params)]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, vhat_t))
            else:
                p_t = m_t / (K.sqrt(v_t) + self.epsilon)

            if self.initial_weight_decay > 0.0:
                if self.weight_decay_pattern is None:
                    p_t += self.weight_decay * p
                else:
                    for pattern in self.weight_decay_pattern:
                        if pattern in p.name:
                            p_t += self.weight_decay * p
                            break
            p_t = p - lr_t * p_t

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {
            'decay_steps': float(K.get_value(self.decay_steps)),
            'warmup_steps': float(K.get_value(self.warmup_steps)),
            'min_lr': float(K.get_value(self.min_lr)),
            'learning_rate': float(K.get_value(self.learning_rate)),
            'beta_1': float(K.get_value(self.beta_1)),
            'beta_2': float(K.get_value(self.beta_2)),
            'epsilon': self.epsilon,
            'weight_decay': float(K.get_value(self.weight_decay)),
            'weight_decay_pattern': self.weight_decay_pattern,
            'amsgrad': self.amsgrad,
        }
        base_config = super(AdamWarmup, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
