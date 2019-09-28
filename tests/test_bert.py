import unittest
import os
import tempfile
import numpy as np
from keras_bert.backend import keras
from keras_bert.backend import backend as K
from keras_bert import (get_model, compile_model, get_base_dict, gen_batch_inputs, get_token_embedding,
                        get_custom_objects, set_custom_objects)


class TestBERT(unittest.TestCase):

    def test_sample(self):
        model = get_model(
            token_num=200,
            head_num=3,
            transformer_num=2,
        )
        model_path = os.path.join(tempfile.gettempdir(), 'keras_bert_%f.h5' % np.random.random())
        model.save(model_path)
        model = keras.models.load_model(
            model_path,
            custom_objects=get_custom_objects(),
        )
        model.summary(line_length=200)

    def test_task_embed(self):
        inputs, outputs = get_model(
            token_num=20,
            embed_dim=12,
            head_num=3,
            transformer_num=2,
            use_task_embed=True,
            task_num=10,
            training=False,
            dropout_rate=0.0,
        )
        model = keras.models.Model(inputs, outputs)
        model_path = os.path.join(tempfile.gettempdir(), 'keras_bert_%f.h5' % np.random.random())
        model.save(model_path)
        model = keras.models.load_model(
            model_path,
            custom_objects=get_custom_objects(),
        )
        model.summary(line_length=200)

    def test_save_load_json(self):
        model = get_model(
            token_num=200,
            head_num=3,
            transformer_num=2,
            attention_activation='gelu',
        )
        compile_model(model)
        data = model.to_json()
        set_custom_objects()
        model = keras.models.model_from_json(data)
        model.summary()

    def test_get_token_embedding(self):
        model = get_model(
            token_num=200,
            head_num=3,
            transformer_num=2,
            attention_activation='gelu',
        )
        embed = get_token_embedding(model)
        self.assertEqual((200, 768), K.int_shape(embed))

    def test_fit(self):
        current_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_path, 'test_bert_fit.h5')
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
        if os.path.exists(model_path):
            steps_per_epoch = 10
            model = keras.models.load_model(
                model_path,
                custom_objects=get_custom_objects(),
            )
        else:
            steps_per_epoch = 1000
            model = get_model(
                token_num=len(token_dict),
                head_num=5,
                transformer_num=12,
                embed_dim=25,
                feed_forward_dim=100,
                seq_len=20,
                pos_num=20,
                dropout_rate=0.05,
                attention_activation='gelu',
            )
            compile_model(
                model,
                learning_rate=1e-3,
                decay_steps=30000,
                warmup_steps=10000,
                weight_decay=1e-3,
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
                    swap_sentence_rate=1.0,
                )

        model.fit_generator(
            generator=_generator(),
            steps_per_epoch=steps_per_epoch,
            epochs=1,
            validation_data=_generator(),
            validation_steps=steps_per_epoch // 10,
        )
        # model.save(model_path)
        for inputs, outputs in _generator():
            predicts = model.predict(inputs)
            outputs = list(map(lambda x: np.squeeze(x, axis=-1), outputs))
            predicts = list(map(lambda x: np.argmax(x, axis=-1), predicts))
            batch_size, seq_len = inputs[-1].shape
            for i in range(batch_size):
                match, total = 0, 0
                for j in range(seq_len):
                    if inputs[-1][i][j]:
                        total += 1
                        if outputs[0][i][j] == predicts[0][i][j]:
                            match += 1
                self.assertGreater(match, total * 0.9)
            self.assertTrue(np.allclose(outputs[1], predicts[1]))
            break
