import tensorflow as tf
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2
from tensorflow.python import ops, math_ops, state_ops, control_flow_ops
from tensorflow.python.keras import backend_config


__all__ = ['AdamWarmup']


class AdamWarmup(OptimizerV2):
    """Adam optimizer with warmup."""

    def __init__(self,
                 decay_steps,
                 warmup_steps,
                 min_lr=0.0,
                 learning_rate=0.001,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-7,
                 weight_decay=0.,
                 weight_decay_pattern=None,
                 amsgrad=False,
                 name='Adam',
                 **kwargs):
        r"""Construct a new Adam optimizer.

        Args:
            decay_steps: Learning rate will decay linearly to zero in decay steps.
            warmup_steps: Learning rate will increase linearly to lr in first warmup steps.
            lr: float >= 0. Learning rate.
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

        super(AdamWarmup, self).__init__(name, **kwargs)
        self._set_hyper('decay_steps', float(decay_steps))
        self._set_hyper('warmup_steps', float(warmup_steps))
        self._set_hyper('min_lr', min_lr)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('decay', self._initial_decay)
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self._set_hyper('weight_decay', weight_decay)
        self.epsilon = epsilon or backend_config.epsilon()
        self.amsgrad = amsgrad
        self._initial_weight_decay = weight_decay
        self._weight_decay_pattern = weight_decay_pattern

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'm')
        for var in var_list:
            self.add_slot(var, 'v')
        if self.amsgrad:
            for var in var_list:
                self.add_slot(var, 'vhat')

    def set_weights(self, weights):
        params = self.weights
        num_vars = int((len(params) - 1) / 2)
        if len(weights) == 3 * num_vars + 1:
            weights = weights[:len(params)]
        super(AdamWarmup, self).set_weights(weights)

    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')
        beta_1_t = self._get_hyper('beta_1', var_dtype)
        beta_2_t = self._get_hyper('beta_2', var_dtype)
        epsilon_t = ops.convert_to_tensor(self.epsilon, var_dtype)
        local_step = math_ops.cast(self.iterations + 1, var_dtype)
        beta_1_power = math_ops.pow(beta_1_t, local_step)
        beta_2_power = math_ops.pow(beta_2_t, local_step)

        decay_steps = self._get_hyper('decay_steps', var_dtype)
        warmup_steps = self._get_hyper('warmup_steps', var_dtype)
        min_lr = self._get_hyper('min_lr', var_dtype)
        lr_t = tf.where(
            local_step <= warmup_steps,
            lr_t * (local_step / warmup_steps),
            min_lr + (lr_t - min_lr) * (1.0 - tf.minimum(local_step, decay_steps) / decay_steps),
        )
        lr_t = (lr_t * math_ops.sqrt(1 - beta_2_power) / (1 - beta_1_power))

        m_t = state_ops.assign(m,
                               beta_1_t * m + (1.0 - beta_1_t) * grad,
                               use_locking=self._use_locking)

        v_t = state_ops.assign(v,
                               beta_2_t * v + (1.0 - beta_2_t) * math_ops.square(grad),
                               use_locking=self._use_locking)

        if self.amsgrad:
            v_hat = self.get_slot(var, 'vhat')
            v_hat_t = math_ops.maximum(v_hat, v_t)
            var_update = m_t / (math_ops.sqrt(v_hat_t) + epsilon_t)
        else:
            var_update = m_t / (math_ops.sqrt(v_t) + epsilon_t)

        if self._initial_weight_decay > 0.0:
            weight_decay = self._get_hyper('weight_decay', var_dtype)
            var_update += weight_decay * var
        var_update = state_ops.assign_sub(var, lr_t * var_update, use_locking=self._use_locking)

        updates = [var_update, m_t, v_t]
        if self.amsgrad:
            updates.append(v_hat_t)
        return control_flow_ops.group(*updates)

    def _resource_apply_sparse(self, grad, var, indices):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        beta_1_t = self._get_hyper('beta_1', var_dtype)
        beta_2_t = self._get_hyper('beta_2', var_dtype)
        epsilon_t = ops.convert_to_tensor(self.epsilon, var_dtype)
        local_step = math_ops.cast(self.iterations + 1, var_dtype)
        beta_1_power = math_ops.pow(beta_1_t, local_step)
        beta_2_power = math_ops.pow(beta_2_t, local_step)

        decay_steps = self._get_hyper('decay_steps', var_dtype)
        warmup_steps = self._get_hyper('warmup_steps', var_dtype)
        min_lr = self._get_hyper('min_lr', var_dtype)
        lr_t = tf.where(
            local_step <= warmup_steps,
            lr_t * (local_step / warmup_steps),
            min_lr + (lr_t - min_lr) * (1.0 - tf.minimum(local_step, decay_steps) / decay_steps),
        )
        lr_t = (lr_t * math_ops.sqrt(1 - beta_2_power) / (1 - beta_1_power))

        m = self.get_slot(var, 'm')
        m_scaled_g_values = grad * (1 - beta_1_t)
        m_t = state_ops.assign(m, m * beta_1_t, use_locking=self._use_locking)
        with ops.control_dependencies([m_t]):
            m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)

        v = self.get_slot(var, 'v')
        v_scaled_g_values = (grad * grad) * (1 - beta_2_t)
        v_t = state_ops.assign(v, v * beta_2_t, use_locking=self._use_locking)
        with ops.control_dependencies([v_t]):
            v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)

        if self.amsgrad:
            v_hat = self.get_slot(var, 'vhat')
            v_hat_t = math_ops.maximum(v_hat, v_t)
            var_update = m_t / (math_ops.sqrt(v_hat_t) + epsilon_t)
        else:
            var_update = m_t / (math_ops.sqrt(v_t) + epsilon_t)

        if self._initial_weight_decay > 0.0:
            weight_decay = self._get_hyper('weight_decay', var_dtype)
            var_update += weight_decay * var
        var_update = state_ops.assign_sub(var, lr_t * var_update, use_locking=self._use_locking)

        updates = [var_update, m_t, v_t]
        if self.amsgrad:
            updates.append(v_hat_t)
        return control_flow_ops.group(*updates)

    def get_config(self):
        config = super(AdamWarmup, self).get_config()
        config.update({
            'decay_steps': self._serialize_hyperparameter('decay_steps'),
            'warmup_steps': self._serialize_hyperparameter('warmup_steps'),
            'min_lr': self._serialize_hyperparameter('min_lr'),
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'decay': self._serialize_hyperparameter('decay'),
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'weight_decay': self._serialize_hyperparameter('weight_decay'),
            'epsilon': self.epsilon,
            'amsgrad': self.amsgrad,
        })
        return config
