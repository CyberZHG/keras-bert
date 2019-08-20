from keras_bert.backend import TF_KERAS

__all__ = ['AdamWarmup', 'calc_train_steps']


def calc_train_steps(num_example, batch_size, epochs, warmup_proportion=0.1):
    """Calculate the number of total and warmup steps.

    >>> calc_train_steps(num_example=1024, batch_size=32, epochs=10, warmup_proportion=0.1)
    (320, 32)

    :param num_example: Number of examples in one epoch.
    :param batch_size: Batch size.
    :param epochs: Number of epochs.
    :param warmup_proportion: The proportion of warmup steps.
    :return: Total steps and warmup steps.
    """
    steps = (num_example + batch_size - 1) // batch_size
    total = steps * epochs
    warmup = int(total * warmup_proportion)
    return total, warmup


if TF_KERAS:
    from .warmup_v2 import AdamWarmup
else:
    from .warmup import AdamWarmup
