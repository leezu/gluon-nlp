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
"""Run evaluation tasks."""

import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import cross_validate

from . import tasks
from .utils import print_time

__all__ = ['run', 'get_tokens', 'cv_classify', 'train_test_classify']


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def train_test_classify(estimator, X_train, Y_train, X_test, Y_test):
    estimator.fit(X_train, Y_train)

    Y_pred = estimator.predict(X_test)
    acc = accuracy_score(Y_test, Y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(Y_test, Y_pred,
                                                       average='micro')

    return dict(precision=prec, recall=rec, f1_score=f1, accuracy=acc)


def cv_classify(estimator, X, Y, cv=10):

    estimator.fit(X, Y)

    scores = cross_validate(
        estimator, X, Y, cv=5, scoring=[
            'accuracy', 'f1_micro', 'precision_micro', 'recall_micro'],
        return_train_score=False)

    return scores


def run(names, emb, logfile=None):
    results = dict()

    for name in names:
        if name == 'affix_prediction':
            with print_time('affix_prediction'):
                estimator = LogisticRegression(C=1e5, solver='lbfgs',
                                               multi_class='multinomial',
                                               verbose=True, max_iter=5000)
                scores = train_test_classify(estimator,
                                             *tasks.affix_prediction(emb))
                print(scores)
                results['affix_prediction'] = scores

        elif name in ['mr', 'subj', 'imdb']:
            with print_time(name):
                estimator = LogisticRegression(C=1e5, solver='lbfgs',
                                               multi_class='multinomial',
                                               verbose=True, max_iter=5000)
                with print_time('load data'):
                    data = tasks.get_mr_subj_imdb(name, emb)
                scores = cv_classify(estimator, *data)
                print(scores)
                results['affix_prediction'] = scores

        elif name in ['pos', 'chunk']:
            with print_time(name):
                estimator = LogisticRegression(C=1e5, solver='saga',
                                               multi_class='multinomial',
                                               max_iter=20, verbose=True)
                with print_time('load data'):
                    data = tasks.get_pos_chunk(name, emb)
                scores = train_test_classify(estimator, *data)
                print(scores)
                results[name] = scores

    if logfile is not None:
        with open(logfile, 'w') as f:
            json.dump(results, f, cls=_NumpyEncoder, indent=2)

    return results


def get_tokens(names):
    tokens = set()
    for name in names:
        if name == 'affix_prediction':
            tokens.update(tasks.affix_prediction())
        elif name in ['mr', 'subj', 'imdb']:
            with print_time(name):
                tokens.update(tasks.get_mr_subj_imdb(name))
        if name in ['pos', 'chunk']:
            with print_time(name):
                tokens.update(tasks.get_pos_chunk(name))
    return tokens
