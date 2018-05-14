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
"""Data helpers"""

import functools
import itertools
import logging
import math
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import gluonnlp as nlp
import subword
import utils

try:
    import ujson as json
except ImportError:
    import json
    logging.warning('ujson not installed. '
                    ' Install via `pip install ujson` '
                    'for faster data preprocessing.')


###############################################################################
# Hyperparameters
###############################################################################
def add_parameters(parser):
    group = parser.add_argument_group('Dataset parameters')
    group.add_argument('--train-dataset', type=str, default='Text8',
                       help='Training corpus. '
                       '[\'Text8\', \'Test\', \'Wikipedia\']')

    # Text8 arguments
    pass

    # Wikipedia arguments
    group.add_argument(
        '--wikipedia-date', type=str, default='auto',
        help='Version of wikipedia dataset to use. '
        'Auto chooses the most recent version.'
        'Manual specification in format YYYYMMDD, e.g. 20180514.')
    group.add_argument('--wikipedia-language', type=str, default='en',
                       help='Language of wikipedia dataset to use. ')


###############################################################################
# Data helpers
###############################################################################
def token_to_index(serialized_sentences, vocab):
    sentences = json.loads(serialized_sentences)
    coded = [
        np.array([vocab[token] for token in s
                  if token in vocab], dtype=np.int32) for s in sentences
    ]
    return coded


def _get_sentence_corpus(args):
    with utils.print_time('read dataset to memory'):
        # TODO(leezu) Remove this Test dataset
        if args.train_dataset.lower() == 'test':
            sentences = nlp.data.Text8(segment='train')
            sentences = [sentences[0][:1000]]
        elif args.train_dataset.lower() == 'text8':
            sentences = nlp.data.Text8(segment='train')
        elif args.train_dataset.lower() == 'wikipedia':
            sentences = nlp.data.Wikipedia(date=args.wikipedia_date,
                                           language=args.wikipedia_language)
        else:
            raise RuntimeError('Unknown dataset.')

    return sentences


def _preprocess_sentences(sentences):
    # Count tokens
    with utils.print_time('count all tokens'):
        counter = nlp.data.count_tokens(
            itertools.chain.from_iterable(sentences))

    vocab = nlp.Vocab(counter, unknown_token=None, padding_token=None,
                      bos_token=None, eos_token=None, min_freq=5)

    # Split dataset into parts for fast multiprocessing (and serialize with
    # json to avoid slow pickle operation)
    num_workers = mp.cpu_count()
    if len(sentences) == 1:
        size = math.ceil(len(sentences[0]) / num_workers)
        worker_sentences = [[sentences[0][i:i + size]]
                            for i in range(0, len(sentences[0]), size)]
    else:
        size = math.ceil(len(sentences) / num_workers)
        worker_sentences = [
            sentences[i:i + size] for i in range(0, len(sentences), size)
        ]

    worker_sentences = [json.dumps(s) for s in worker_sentences]
    with mp.Pool(processes=num_workers) as pool:
        with utils.print_time('code all sentences'):
            coded = pool.map(
                functools.partial(token_to_index, vocab=vocab),
                worker_sentences)
            coded = sum(coded, [])
            if len(sentences) == 1:
                coded = [np.concatenate(coded)]

    # Prune frequent words from sentences
    with utils.print_time('prune frequent words from sentences'):
        frequent_tokens_subsampling_constant = 1e-3
        idx_to_counts = np.array(vocab.idx_to_counts, dtype=int)
        f = idx_to_counts / np.sum(idx_to_counts)
        idx_to_pdiscard = (np.sqrt(frequent_tokens_subsampling_constant / f) +
                           frequent_tokens_subsampling_constant / f)

        # prune_sentences releases GIL so multi-threading is sufficient
        prune_sentences = functools.partial(
            nlp.data.word_embedding_training.prune_sentences,
            idx_to_pdiscard=idx_to_pdiscard)
        with ThreadPoolExecutor(max_workers=num_workers) as e:
            coded = list(e.map(prune_sentences, coded))

    return vocab, coded


def _get_train_dataset(args, vocab, coded):
    idx_to_counts = np.array(vocab.idx_to_counts, dtype=int)
    # Get index to byte mapping from vocab

    if args.objective.lower() == 'skipgram':
        # TODO: Enable SkipGramWordEmbeddingDataset without subword_vocab
        # if not args.subword_network.lower():
        #     subword_vocab = None
        #     dataset = nlp.data.SkipGramWordEmbeddingDataset(
        #         coded, idx_to_counts)
        if args.subword_network.lower() != 'fasttext':
            subword_function = nlp.vocab.create('ByteSubwords')
            subword_vocab = nlp.SubwordVocab(idx_to_token=vocab.idx_to_token,
                                             subword_function=subword_function,
                                             merge_indices=False)

            # Get subword network data requirements
            if args.subword_network:
                min_size = subword.list_subwordnetworks(
                    args.subword_network).min_size
            else:
                min_size = 1
            dataset = nlp.data.SkipGramWordEmbeddingDataset(
                coded, idx_to_counts, subword_vocab, min_size=min_size)
            # TODO: Enable SkipGramWordEmbeddingDataset without subword_vocab
            # As a workaround, set subword_vocab to None here
            if not args.subword_network.lower():
                subword_vocab = None
        else:  # Optimized fasttext data iterator for fasttext
            subword_function = nlp.vocab.create(
                'NGramSubwords', vocabulary=vocab, ngrams=[3, 4, 5, 6],
                max_num_subwords=1000000)
            subword_vocab = nlp.SubwordVocab(idx_to_token=vocab.idx_to_token,
                                             subword_function=subword_function,
                                             merge_indices=True)
            dataset = nlp.data.SkipGramFasttextWordEmbeddingDataset(
                coded, idx_to_counts, subword_vocab)
    else:
        raise NotImplementedError('Objective {} not implemented.'.format(
            args.objective))

    return dataset, vocab, subword_vocab


def get_train_data(args):
    # TODO currently only supports skipgram
    sentences = _get_sentence_corpus(args)
    vocab, coded = _preprocess_sentences(sentences)
    data = _get_train_dataset(args, vocab, coded)
    return data
