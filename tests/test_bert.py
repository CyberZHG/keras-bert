import unittest
import os
import tempfile
import random
import keras
from keras_bert import get_model, get_custom_objects, get_base_dict, gen_batch_inputs


class TestBERT(unittest.TestCase):

    def test_sample(self):
        model = get_model(
            token_num=200,
            head_num=3,
            transformer_num=2,
        )
        model_path = os.path.join(tempfile.gettempdir(), 'keras_bert_%f.h5' % random.random())
        model.save(model_path)
        model = keras.models.load_model(
            model_path,
            custom_objects=get_custom_objects(),
        )
        model.summary(line_length=200)

    def test_fit(self):
        sentence_pairs = [
            [['all', 'work', 'and', 'no', 'play'], ['makes', 'jack', 'a', 'dull', 'boy']],
            [['from', 'the', 'day', 'forth'], ['my', 'arm', 'changed']],
            [['and', 'a', 'voice', 'echoed'], ['power', 'give', 'me', 'more', 'power']],
        ]
        token_dict = get_base_dict()
        for pairs in sentence_pairs:
            for token in pairs[0] + pairs[1]:
                if token not in token_dict:
                    token_dict[token] = len(token_dict)
        token_list = list(token_dict.keys())
        model = get_model(
            token_num=len(token_dict),
            head_num=2,
            transformer_num=2,
            embed_dim=10,
            feed_forward_dim=40,
            seq_len=20,
            pos_num=20,
            dropout=0.1,
        )
        model.summary()

        def _generator():
            while True:
                yield gen_batch_inputs(
                    sentence_pairs,
                    token_dict,
                    token_list,
                    seq_len=20,
                    mask_rate=0.3,
                )

        model.fit_generator(
            generator=_generator(),
            steps_per_epoch=1000,
            epochs=30,
            validation_data=_generator(),
            validation_steps=100,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
            ],
        )
