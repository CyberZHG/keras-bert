import keras


def get_inputs(seq_len, use_nsp=True):
    """Get input layers.

    :param seq_len: Length of the sequence or None.
    :param use_nsp: Whether to use next sentence prediction.
    :return: Token input, (segment input,) position input and masked positions.
    """
    names = ['Token', 'Position', 'Masked']
    if use_nsp:
        names.insert(1, 'Segment')
    return [keras.layers.Input(
        shape=(seq_len,),
        name='Input-%s' % name,
    ) for name in names]
