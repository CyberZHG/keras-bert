import keras


def get_inputs(seq_len):
    """Get input layers.

    :param seq_len: Length of the sequence or None.
    """
    names = ['Token', 'Segment', 'Position', 'Masked']
    return [keras.layers.Input(
        shape=(seq_len,),
        name='Input-%s' % name,
    ) for name in names]
