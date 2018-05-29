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
"""Plots"""

import itertools
import statistics
import random
import math

import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

sns.set(color_codes=True)

import arguments
import data

import gluonnlp as nlp


def add_parameters(parser):
    group = parser.add_argument_group('Plotting (run plot.py)')
    group.add_argument('--plot-vocab', type=str, default='',
                       help='Path to vocab file.')
    group.add_argument('--plot-eval-words', type=str, default='')


def make_plot(vocab, observed_words, name, subsample=None):
    print('Creating word frequency plot and distplot of evaluation words.')
    counts = np.array(vocab.idx_to_counts)
    count_df = pd.DataFrame(dict(word=vocab.idx_to_token, y=counts))

    if subsample is None:
        print('Observed ', len(observed_words), 'words. Drawing all')
        a = observed_words
    else:
        print('Observed ', len(observed_words), 'words. Drawing ', subsample)
        a = random.choices(observed_words, k=5000)

    ax = count_df.y.plot()
    ax = sns.distplot(a=a, ax=ax, hist=False, kde=False, rug=True)
    ax.set(yscale="log")
    ax.figure.savefig(f'{name}.png')
    plt.close(ax.figure)

    ax = count_df.y.plot()
    ax = sns.distplot(a=a, ax=ax, hist=False, kde=False, rug=True)
    ax.set(xscale="log", yscale="log")
    ax.figure.savefig(f'loglog_{name}.png')
    plt.close(ax.figure)


if __name__ == '__main__':
    args = arguments.get_and_setup()

    if args.plot_vocab:
        print('Reading vocab from ', args.plot_vocab)
        with open(args.plot_vocab, 'r') as f:
            vocab = nlp.Vocab.from_json(f.read())
    else:
        print('Computing training vocab')
        _, vocab, _ = data.get_train_data(args)

    print(vocab)

    dataset_stats = []
    observed_words = []

    if args.plot_eval_words.lower() == 'similarity':
        similarity_datasets = nlp.data.word_embedding_evaluation.\
            word_similarity_datasets
        for dataset_name in similarity_datasets:
            parameters = nlp.data.list_datasets(dataset_name)
            for key_values in itertools.product(*parameters.values()):
                dataset_kwargs = dict(zip(parameters.keys(), key_values))
                dataset = nlp.data.create(dataset_name, **dataset_kwargs)

                dataset_idxs = [
                    vocab.token_to_idx[d[0].lower()] for d in dataset
                    if d[0] in vocab.token_to_idx
                ] + [
                    vocab.token_to_idx[d[1].lower()] for d in dataset
                    if d[1] in vocab.token_to_idx
                ]
                observed_words += dataset_idxs
                oov = sum(d[0].lower() not in vocab.token_to_idx
                          for d in dataset) + sum(
                              d[1].lower() not in vocab.token_to_idx
                              for d in dataset)
                non_oov = len(dataset_idxs)

                dataset_stats.append(
                    dict(name=dataset_name, kwargs=dataset_kwargs,
                         mean_word=statistics.mean(dataset_idxs),
                         mean_word_pos=statistics.mean(dataset_idxs) /
                         len(vocab), mean_word_freq=vocab.idx_to_counts[int(
                             statistics.mean(dataset_idxs))], oov=oov,
                         non_oov=non_oov,
                         stdv_word=statistics.stdev(dataset_idxs)))

                print(dataset_stats[-1])
                name = dataset_name + str(list(
                    dataset_kwargs.values())[0]) if list(
                        dataset_kwargs.values()) else dataset_name
                make_plot(vocab, dataset_idxs, name)

    elif args.plot_eval_words.lower() == 'analogy':
        analogy_datasets = nlp.data.word_embedding_evaluation.\
            word_analogy_datasets
        for dataset_name in analogy_datasets:
            parameters = nlp.data.list_datasets(dataset_name)
            for key_values in itertools.product(*parameters.values()):
                dataset_kwargs = dict(zip(parameters.keys(), key_values))
                dataset = nlp.data.create(dataset_name, **dataset_kwargs)

                dataset_idxs = [
                    vocab.token_to_idx[d[0].lower()] for d in dataset
                    if d[0] in vocab.token_to_idx
                ] + [
                    vocab.token_to_idx[d[1].lower()] for d in dataset
                    if d[1] in vocab.token_to_idx
                ] + [
                    vocab.token_to_idx[d[2].lower()] for d in dataset
                    if d[2] in vocab.token_to_idx
                ] + [
                    vocab.token_to_idx[d[3].lower()] for d in dataset
                    if d[3] in vocab.token_to_idx
                ]
                observed_words += dataset_idxs
                oov = sum(d[0].lower() not in vocab.token_to_idx
                          for d in dataset) + sum(
                              d[1].lower() not in vocab.token_to_idx
                              for d in dataset) + sum(
                                  d[2].lower() not in vocab.token_to_idx
                                  for d in dataset) + sum(
                                      d[3].lower() not in vocab.token_to_idx
                                      for d in dataset)
                non_oov = len(dataset_idxs)

                dataset_stats.append(
                    dict(name=dataset_name, kwargs=dataset_kwargs,
                         mean_word=statistics.mean(dataset_idxs),
                         mean_word_pos=statistics.mean(dataset_idxs) /
                         len(vocab), mean_word_freq=vocab.idx_to_counts[int(
                             statistics.mean(dataset_idxs))], oov=oov,
                         non_oov=non_oov,
                         stdv_word=statistics.stdev(dataset_idxs)))

                print(dataset_stats[-1])

    make_plot(vocab, observed_words, 'all', subsample=5000)
