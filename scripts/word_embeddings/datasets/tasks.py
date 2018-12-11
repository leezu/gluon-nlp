# coding: utf-8

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Evaluation tasks."""
import re
import itertools

import mxnet as mx
import numpy as np
import sklearn

import gluonnlp as nlp

from .morphology import LazaridouDerivationalMorphologyDataset
from .utils import sliding_window_view

__all__ = ['affix_prediction', 'get_pos_chunk', 'get_mr_subj_imdb']

# TODO improve the data cleaning


def _clean_str_sst12(string):
    string = re.sub(r'[^A-Za-z0-9(),!?\'\`]', ' ', string)
    string = re.sub(r'\s{2,}', ' ', string)
    return string.strip().lower()


def _clean_str(string):
    string = re.sub(r'[^A-Za-z0-9(),!?\'\`]', ' ', string)
    string = re.sub(r'\'s', ' \'s', string)
    string = re.sub(r'\'ve', ' \'ve', string)
    string = re.sub(r'n\'t', ' n\'t', string)
    string = re.sub(r'\'re', ' \'re', string)
    string = re.sub(r'\'d', ' \'d', string)
    string = re.sub(r'\'ll', ' \'ll', string)
    string = re.sub(r',', ' , ', string)
    string = re.sub(r'!', ' ! ', string)
    string = re.sub(r'\(', ' ( ', string)
    string = re.sub(r'\)', ' ) ', string)
    string = re.sub(r'\?', ' ? ', string)
    string = re.sub(r'\s{2,}', ' ', string)
    return string.strip().lower()


def affix_prediction(token_embedding=None):
    # TODO while related work uses train-test split, better to use CV
    def get_X(dataset):
        """Get X for the Affix Prediction task."""
        derived_idx = dataset.header.index('derived')
        X = [sample[derived_idx] for sample in dataset]
        return X

    def get_Y(dataset):
        """Get Y for the Affix Prediction task."""
        affix_idx = dataset.header.index('affix')
        Y = [sample[affix_idx] for sample in dataset]
        affixes = sorted(dataset.affixes)
        affix_to_idx = {a: i for i, a in enumerate(affixes)}
        Y = [affix_to_idx[affix] for affix in Y]
        return Y

    d = [LazaridouDerivationalMorphologyDataset(d) for d in ('train', 'test')]

    if token_embedding is not None:
        X = [token_embedding[get_X(d_)].asnumpy() for d_ in d]
        X_train, X_test = [sklearn.preprocessing.normalize(x) for x in X]
        Y_train, Y_test = [get_Y(d_) for d_ in d]
        return X_train, Y_train, X_test, Y_test
    else:
        derived_idx = d[0].header.index('derived')
        sentences = (x[derived_idx] for d_ in d for x in d_)
        return set(itertools.chain.from_iterable(sentences))


def get_pos_chunk(name, token_embedding=None, padding_token='<pad>'):
    data = [nlp.data.CoNLL2000(t) for t in ('train', 'test')]

    assert name in ('pos', 'chunk')
    y_idx = 1 if name == 'pos' else 2

    def window(sentence, ws=2):
        assert len(sentence)
        sentence = [padding_token] * ws + sentence + [padding_token] * ws
        embs = token_embedding[sentence].asnumpy()
        embs = sklearn.preprocessing.normalize(embs)
        windowed = sliding_window_view(embs, (5, 1))
        # Remove last axis (shape 1) via reshape and swap
        windowed = windowed.reshape((-1, 300, 5)).swapaxes(1, 2)
        # Merge all embedding vectors in the window (sklearn expects 1D)
        windowed = windowed.reshape((-1, 300 * 5))
        return windowed

    if token_embedding is not None:
        X = [np.concatenate([window(s[0]) for s in d]) for d in data]
        Y = [np.concatenate([sample[y_idx] for sample in d]) for d in data]

        X_train, X_test = X
        Y_train, Y_test = Y

        return X_train, Y_train, X_test, Y_test
    else:
        sentences = (x[0] for d in data for x in d)
        return set(itertools.chain.from_iterable(sentences))


def get_mr_subj_imdb(name, token_embedding=None):
    if name.lower() == 'mr':
        data = nlp.data.MR()
    elif name.lower() == 'subj':
        data = nlp.data.SUBJ()
    elif name.lower() == 'imdb':
        data = nlp.data.IMDB()

    X = [_clean_str(x[0]).split() for x in data]
    if token_embedding is not None:
        X = np.stack([
            sklearn.preprocessing.normalize(
                token_embedding[x].asnumpy()).sum(axis=0) / np.sqrt(len(x))
            for x in X])
        Y = np.array([x[1] for x in data])

        return X, Y
    else:
        return set(itertools.chain.from_iterable(X))
