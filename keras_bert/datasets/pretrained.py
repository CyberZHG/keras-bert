# coding=utf-8
from __future__ import unicode_literals

import os
import shutil
from collections import namedtuple
from keras_bert.backend import keras

__all__ = ['PretrainedInfo', 'PretrainedList', 'get_pretrained']


PretrainedInfo = namedtuple('PretrainedInfo', ['url', 'extract_name', 'target_name'])


class PretrainedList(object):

    __test__ = PretrainedInfo(
        'https://github.com/CyberZHG/keras-bert/archive/master.zip',
        'keras-bert-master',
        'keras-bert',
    )

    multi_cased_base = 'https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip'
    chinese_base = 'https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip'
    wwm_uncased_large = 'https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip'
    wwm_cased_large = 'https://storage.googleapis.com/bert_models/2019_05_30/wwm_cased_L-24_H-1024_A-16.zip'
    chinese_wwm_base = PretrainedInfo(
        'https://storage.googleapis.com/hfl-rc/chinese-bert/chinese_wwm_L-12_H-768_A-12.zip',
        'publish',
        'chinese_wwm_L-12_H-768_A-12',
    )


def get_pretrained(info):
    path = info
    if isinstance(info, PretrainedInfo):
        path = info.url
    path = keras.utils.get_file(fname=os.path.split(path)[-1], origin=path, extract=True)
    base_part, file_part = os.path.split(path)
    file_part = file_part.split('.')[0]
    if isinstance(info, PretrainedInfo):
        extract_path = os.path.join(base_part, info.extract_name)
        target_path = os.path.join(base_part, info.target_name)
        if not os.path.exists(target_path):
            shutil.move(extract_path, target_path)
        file_part = info.target_name
    return os.path.join(base_part, file_part)
