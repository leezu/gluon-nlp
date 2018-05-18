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
import collections
import itertools
import logging
import math
import multiprocessing as mp
import concurrent.futures
import os

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

    # Wikipedia arguments
    group.add_argument(
        '--wikipedia-date', type=str, default='auto',
        help='Version of wikipedia dataset to use. '
        'Auto chooses the most recent version.'
        'Manual specification in format YYYYMMDD, e.g. 20180514.')
    group.add_argument('--wikipedia-language', type=str, default='en',
                       help='Language of wikipedia dataset to use. ')
    group.add_argument('--wikipedia-min-sentence-length', type=int, default=5,
                       help='Minimum number of tokens to keep a sentence.')
    group.add_argument('--wikipedia-min-worker-tok-freq', type=int, default=5,
                       help='Tokens that occur less frequently are dropped. '
                       'The threshold is applied per worker process.')
    group.add_argument('--wikipedia-num-parts', type=int, default=100,
                       help='Number of shards to read. Maximum: 100')
    group.add_argument('--wikipedia-vocab-max-token-length', type=int,
                       default=20, help='Filter tokens that are longer.')


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
        elif args.train_dataset.lower() == 'wikitext2':
            sentences = nlp.data.WikiText2(segment='train')
        elif args.train_dataset.lower() == 'wikitext103':
            sentences = nlp.data.WikiText103(segment='train')
        elif args.train_dataset.lower() == 'wikipedia':
            raise RuntimeError(
                'Use map reduce function for wikipedia dataset.')
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
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_workers) as e:
            coded = list(e.map(prune_sentences, coded))

    return vocab, coded


###############################################################################
# Map reduce data helpers
###############################################################################
def _counter(dataset_class, args, **kwargs):
    sentences = dataset_class(**kwargs)
    counts = nlp.data.utils.count_tokens(
        itertools.chain.from_iterable(sentences))
    for key, count in counts.most_common():
        if len(key) > args.wikipedia_vocab_max_token_length:
            del counts[key]
        elif count < args.wikipedia_min_worker_tok_freq:
            del counts[key]
    return counts


def _map_counter(dataset_class, kwargs, num_files, args):
    with concurrent.futures.ProcessPoolExecutor() as e:
        for i in range(num_files):
            future = e.submit(_counter, i=i, dataset_class=dataset_class,
                              args=args, **kwargs)
            yield future


def _reduce_counter(futures):
    future_list = list(futures)
    future_seq = concurrent.futures.as_completed(future_list)
    counts = sum((f.result() for f in future_seq), collections.Counter())
    return counts


def _coder(dataset_class, vocab, min_length, idx_to_pdiscard, **kwargs):
    sentences = dataset_class(**kwargs)
    prune_sentences_ = functools.partial(
        nlp.data.word_embedding_training.prune_sentences,
        idx_to_pdiscard=idx_to_pdiscard)
    coded = [
        prune_sentences_(
            np.array([vocab[token] for token in s
                      if token in vocab], dtype=np.int32)) for s in sentences
        if len(s) > min_length
    ]
    # Pruning shortens, throw away too short sentences
    coded = [c for c in coded if len(c) > min_length]
    sentence_boundaries = np.cumsum([len(c) for c in coded])
    coded = np.concatenate(coded)
    return sentence_boundaries, coded


def _map_coder(dataset_class, kwargs, num_files, vocab, min_length,
               idx_to_pdiscard):
    with concurrent.futures.ProcessPoolExecutor() as e:
        for i in range(num_files):
            future = e.submit(_coder, i=i, dataset_class=dataset_class,
                              vocab=vocab, min_length=min_length,
                              idx_to_pdiscard=idx_to_pdiscard, **kwargs)
            yield future


def _reduce_coder(futures):
    future_list = list(futures)
    future_seq = concurrent.futures.as_completed(future_list)
    all_sentence_boundaries = []
    all_coded = []
    for f in future_seq:
        sentence_boundaries, coded = f.result()
        # Correct sentence boundaries
        if len(all_sentence_boundaries):
            sentence_boundaries = sentence_boundaries + \
                all_sentence_boundaries[-1][-1]
        all_sentence_boundaries.append(sentence_boundaries)
        all_coded.append(coded)
    sentence_boundaries = np.concatenate(all_sentence_boundaries)
    coded = np.concatenate(all_coded)
    return sentence_boundaries, coded


def _preprocess_sentences_map_reduce(args):
    if args.train_dataset.lower() == 'wikipedia':
        dataset_class = nlp.data.Wikipedia
        kwargs = dict(date=args.wikipedia_date,
                      language=args.wikipedia_language)
    else:
        raise RuntimeError('Unsupported dataset')

    num_files = min(args.wikipedia_num_parts, dataset_class.num_files)

    with utils.print_time('read dataset and count tokens'):
        counter_futures = _map_counter(dataset_class, kwargs, num_files, args)
        counter = _reduce_counter(counter_futures)

    vocab = nlp.Vocab(counter, unknown_token=None, padding_token=None,
                      bos_token=None, eos_token=None, min_freq=5,
                      max_token_length=args.wikipedia_vocab_max_token_length)
    # Prepare datastructures for pruning frequent tokens
    frequent_tokens_subsampling_constant = 1e-3
    idx_to_counts = np.array(vocab.idx_to_counts, dtype=int)
    f = idx_to_counts / np.sum(idx_to_counts)
    idx_to_pdiscard = (np.sqrt(frequent_tokens_subsampling_constant / f) +
                       frequent_tokens_subsampling_constant / f)

    with utils.print_time('read dataset and code'):
        coded_futures = _map_coder(dataset_class, kwargs, num_files, vocab,
                                   args.wikipedia_min_sentence_length,
                                   idx_to_pdiscard)
        sentence_boundaries, coded = _reduce_coder(coded_futures)

    return vocab, sentence_boundaries, coded


def _get_train_dataset(args, vocab, coded, sentence_boundaries=None):
    idx_to_counts = np.array(vocab.idx_to_counts, dtype=int)
    # Get index to byte mapping from vocab
    if args.objective.lower() == 'skipgram':
        # TODO don't create subword vocab when not needed
        with utils.print_time('create subword vocabulary'):
            if args.subword_function.lower() == 'byte':
                subword_function = nlp.vocab.create('ByteSubwords')
            elif args.subword_function.lower() == 'character':
                subword_function = nlp.vocab.create('CharacterSubwords',
                                                    vocabulary=vocab)
            elif args.subword_function.lower() == 'ngrams':
                subword_function = nlp.vocab.create(
                    'NGramSubwords', vocabulary=vocab, ngrams=[3, 4, 5, 6],
                    max_num_subwords=1000000)
            subword_vocab = nlp.SubwordVocab(idx_to_token=vocab.idx_to_token,
                                             subword_function=subword_function,
                                             merge_indices=False)

        # Get subword network data requirements
        if args.subword_network:
            min_size = subword.list_subwordnetworks(
                args.subword_network).min_size
        else:
            min_size = 1

        with utils.print_time('create skipgram dataset'):
            dataset = nlp.data.SkipGramWordEmbeddingDataset(
                coded=coded, idx_to_counts=idx_to_counts,
                subword_vocab=subword_vocab, min_size=min_size,
                sentence_boundaries=sentence_boundaries)

        # TODO: Enable SkipGramWordEmbeddingDataset without subword_vocab
        # As a workaround, set subword_vocab to None here
        if not args.subword_network:
            subword_vocab = None
    else:
        raise NotImplementedError('Objective {} not implemented.'.format(
            args.objective))

    return dataset, vocab, subword_vocab


def get_train_data(args):
    # TODO currently only supports skipgram
    if args.train_dataset.lower() != 'wikipedia':
        sentences = _get_sentence_corpus(args)
        vocab, coded = _preprocess_sentences(sentences)
        sentence_boundaries = None
    else:
        vocab, sentence_boundaries, coded = _preprocess_sentences_map_reduce(
            args)
    dataset, vocab, subword_vocab = _get_train_dataset(
        args, vocab, coded, sentence_boundaries=sentence_boundaries)

    # Log vocab and subword vocab
    vocab.to_json(os.path.join(args.logdir, 'vocab.json'))
    if subword_vocab is not None:
        subword_vocab.to_json(os.path.join(args.logdir, 'subword_vocab.json'))

    return dataset, vocab, subword_vocab
