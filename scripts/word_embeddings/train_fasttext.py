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

# pylint: disable=global-variable-undefined,wrong-import-position
"""Fasttext embedding model
===========================

This example shows how to train a FastText embedding model on Text8 with the
Gluon NLP Toolkit.

The FastText embedding model was introduced by

- Bojanowski, P., Grave, E., Joulin, A., & Mikolov, T. (2017). Enriching word
  vectors with subword information. TACL, 5(), 135–146.

When setting --ngram-buckets to 0, a Word2Vec embedding model is trained. The
Word2Vec embedding model was introduced by

- Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient estimation
  of word representations in vector space. ICLR Workshop , 2013

"""
# Set a few mxnet specific environment variables
import os
# Workaround for https://github.com/apache/incubator-mxnet/issues/11314
os.environ['MXNET_FORCE_ADDTAKEGRAD'] = '1'
os.environ['MXNET_GPU_MEM_POOL_TYPE'] = 'Round'

import argparse
import itertools
import logging
import random
import math
import sys
import tempfile
import time
import warnings

import mxnet as mx
import numpy as np
import gluonnlp as nlp
from gluonnlp.base import numba_njit, numba_prange

import trainer
import evaluation
from data import WikiDumpStream
from candidate_sampler import remove_accidental_hits
from utils import get_context, print_time


###############################################################################
# Utils
###############################################################################
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Word embedding training with Gluon.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Data options
    group = parser.add_argument_group('Data arguments')
    group.add_argument('--data', type=str, default='text8',
                       help='Training dataset.')
    group.add_argument('--wiki-root', type=str, default='text8',
                       help='Root under which preprocessed wiki dump.')
    group.add_argument('--wiki-language', type=str, default='text8',
                       help='Language of wiki dump.')
    group.add_argument('--wiki-date', help='Date of wiki dump.')
    group.add_argument('--max-words', type=int,
                       help='Maximum number of words to train on.')

    # Computation options
    group = parser.add_argument_group('Computation arguments')
    group.add_argument('--batch-size', type=int, default=2048,
                       help='Batch size for training.')
    group.add_argument('--epochs', type=int, default=5, help='Epoch limit')
    group.add_argument('--gpu', type=int, nargs='+',
                       help=('Number (index) of GPU to run on, e.g. 0. '
                             'If not specified, uses CPU.'))
    group.add_argument('--no-hybridize', action='store_true',
                       help='Disable hybridization of gluon HybridBlocks.')
    group.add_argument(
        '--no-static-alloc', action='store_true',
        help='Disable static memory allocation for HybridBlocks.')
    group.add_argument('--no-sparse-grad', action='store_true',
                       help='Disable sparse gradient support.')
    group.add_argument('--no-lazy-update', action='store_true',
                       help='Disable lazy parameter update for sparse gradient.')

    # Model
    group = parser.add_argument_group('Model arguments')
    group.add_argument('--emsize', type=int, default=300,
                       help='Size of embedding vectors.')
    group.add_argument('--subword-network', type=str, default='fasttext')
    group.add_argument('--ngrams', type=int, nargs='+', default=[3, 4, 5, 6])
    group.add_argument(
        '--ngram-buckets', type=int, default=2000000,
        help='Size of word_context set of the ngram hash function. '
        'Set this to 0 for Word2Vec style training.')
    group.add_argument('--model', type=str, default='skipgram',
                       help='SkipGram or CBOW.')
    group.add_argument('--window', type=int, default=5,
                       help='Context window size.')
    group.add_argument('--negative', type=int, default=5,
                       help='Number of negative samples '
                       'per source-context word pair.')
    group.add_argument('--frequent-token-subsampling', type=float,
                       default=1E-4,
                       help='Frequent token subsampling constant.')
    group.add_argument('--seed', type=int, default=1, help='random seed')

    # Logging
    group = parser.add_argument_group('Logging arguments')
    group.add_argument('--logdir', type=str, default='logs',
                       help='Directory to store logs.')
    group.add_argument('--log-interval', type=int, default=100)
    group.add_argument('--eval-interval', type=int,
                       help='Evaluate every --eval-interval iterations '
                       'in addition to at the end of every epoch.')
    group.add_argument('--no-eval-analogy', action='store_true',
                       help='Don\'t evaluate on the analogy task.')

    # Evaluation options
    evaluation.add_parameters(parser)
    trainer.add_parameters(parser)

    args = parser.parse_args()
    evaluation.validate_args(args)

    random.seed(args.seed)
    mx.random.seed(args.seed)
    np.random.seed(args.seed)

    return args


def get_train_data(args):
    """Helper function to get training data."""

    def text8():
        data = nlp.data.Text8(segment='train')
        counter = nlp.data.count_tokens(itertools.chain.from_iterable(data))
        vocab = nlp.Vocab(counter, unknown_token=None, padding_token=None,
                          bos_token=None, eos_token=None, min_freq=5)
        idx_to_counts = [counter[w] for w in vocab.idx_to_token]
        data = nlp.data.SimpleDataStream([data])
        return data, vocab, idx_to_counts

    def wiki():
        data = WikiDumpStream(
            root=os.path.expanduser(args.wiki_root),
            language=args.wiki_language, date=args.wiki_date)
        vocab = data.vocab
        idx_to_counts = data.idx_to_counts
        return data, vocab, idx_to_counts

    with print_time('load training data'):
        f_data = text8 if args.data == 'text8' else wiki
        data, vocab, idx_to_counts = f_data()

    # Apply transforms
    def code(shard):
        with print_time('code shard'):
            return [[vocab[token] for token in sentence if token in vocab]
                    for sentence in shard]

    def shuffle(shard):
        random.shuffle(shard)
        return shard

    data = data.transform(code)
    data = data.transform(shuffle)

    negatives_sampler = nlp.data.UnigramCandidateSampler(
        weights=mx.nd.array(idx_to_counts)**0.75)

    sum_counts = sum(idx_to_counts)
    idx_to_pdiscard = [
        1 - math.sqrt(args.frequent_token_subsampling / (count / sum_counts))
        for count in idx_to_counts
    ]

    if args.subword_network.lower() == 'fasttext':
        with print_time('prepare subwords'):
            subword_function = nlp.vocab.create_subword_function(
                'NGramHashes', ngrams=args.ngrams,
                num_subwords=args.ngram_buckets)

            # Store subword indices for all words in vocabulary
            idx_to_subwordidxs = list(subword_function(vocab.idx_to_token))
            get_subwords_masks = get_subwords_masks_factory(idx_to_subwordidxs)
            max_subwordidxs_len = max(len(s) for s in idx_to_subwordidxs)
            if max_subwordidxs_len > 500:
                warnings.warn(
                    'The word with largest number of subwords '
                    'has {} subwords, suggesting there are '
                    'some noisy words in your vocabulary. '
                    'You should filter out very long words '
                    'to avoid memory issues.'.format(max_subwordidxs_len))

        return (data, negatives_sampler, vocab, subword_function,
                get_subwords_masks, idx_to_pdiscard, sum_counts,
                idx_to_subwordidxs)
    elif args.subword_network.lower() == 'highwaycnn':
        with print_time('prepare characters / byte mapping'):
            subword_function = nlp.vocab.create_subword_function(
                'ByteSubwords')

            # Store subword indices for all words in vocabulary
            idx_to_subwordidxs = list(subword_function(vocab.idx_to_token))
            get_subwords_masks = get_subwords_masks_factory(idx_to_subwordidxs)
            max_subwordidxs_len = max(len(s) for s in idx_to_subwordidxs)

        return (data, negatives_sampler, vocab, subword_function,
                get_subwords_masks, idx_to_pdiscard, sum_counts,
                idx_to_subwordidxs)
    else:
        return data, negatives_sampler, vocab, idx_to_pdiscard, sum_counts


def get_subwords_masks_factory(idx_to_subwordidxs):
    idx_to_subwordidxs = [
        np.array(i, dtype=np.int_) for i in idx_to_subwordidxs
    ]

    def get_subwords_masks(indices, minlength=1):
        subwords = [idx_to_subwordidxs[i] for i in indices]
        return _get_subwords_masks(subwords, minlength)

    return get_subwords_masks


@numba_njit
def _get_subwords_masks(subwords, minlength):
    lengths = np.array([len(s) for s in subwords])
    length = np.max(lengths)
    length = np.max(np.array([length, minlength]))
    subwords_arr = np.zeros((len(subwords), length))
    mask = np.zeros((len(subwords), length))
    for i in numba_prange(len(subwords)):
        s = subwords[i]
        subwords_arr[i, :len(s)] = s
        mask[i, :len(s)] = 1
    return subwords_arr, mask


def save(args, embedding, embedding_out, vocab):
    f, path = tempfile.mkstemp(dir=args.logdir)
    os.close(f)

    # save vocab
    with open(path, 'w') as f:
        f.write(vocab.to_json())
    os.replace(path, os.path.join(args.logdir, 'vocab.json'))

    # save list of words with zero word vectors
    zero_word_vectors_words = sorted([
        vocab.idx_to_token[idx] for idx in np.where((
            embedding.embedding.weight.data().norm(axis=1) < 1E-5
        ).asnumpy())[0]
    ])
    with open(path, 'w') as f:
        f.write('\n'.join(zero_word_vectors_words))
    os.replace(path, os.path.join(args.logdir, 'zero_word_vectors_words.txt'))

    # write to temporary file; use os.replace
    embedding.collect_params().save(path)
    os.replace(path, os.path.join(args.logdir, 'embedding.params'))
    embedding_out.collect_params().save(path)
    os.replace(path, os.path.join(args.logdir, 'embedding_out.params'))


###############################################################################
# Training code
###############################################################################
def train(args):
    """Training helper."""
    if args.subword_network.lower() == 'fasttext':
        data, negatives_sampler, vocab, subword_function, \
            get_subwords_masks, idx_to_pdiscard, num_tokens, \
            idx_to_subwordidxs = get_train_data(args)
        embedding = nlp.model.train.FasttextEmbeddingModel(
            token_to_idx=vocab.token_to_idx,
            subword_function=subword_function,
            embedding_size=args.emsize,
            weight_initializer=mx.init.Uniform(scale=1 / args.emsize),
            sparse_grad=not args.no_sparse_grad,
        )
    elif args.subword_network.lower() == 'highwaycnn':
        import blocks
        data, negatives_sampler, vocab, subword_function, \
            get_subwords_masks, idx_to_pdiscard, num_tokens, \
            idx_to_subwordidxs = get_train_data(args)
        embedding = blocks.HighwayCNNEmbeddingModel(
            token_to_idx=vocab.token_to_idx,
            subword_function=subword_function,
            embedding_size=args.emsize,
            character_embedding_size=15,
            sparse_grad=not args.no_sparse_grad,
        )
    else:
        data, negatives_sampler, vocab, \
            idx_to_pdiscard, num_tokens = get_train_data(args)
        embedding = nlp.model.train.SimpleEmbeddingModel(
            token_to_idx=vocab.token_to_idx,
            embedding_size=args.emsize,
            weight_initializer=mx.init.Uniform(scale=1 / args.emsize),
            sparse_grad=not args.no_sparse_grad,
        )
    embedding_out = nlp.model.train.SimpleEmbeddingModel(
        token_to_idx=vocab.token_to_idx,
        embedding_size=args.emsize,
        weight_initializer=mx.init.Zero(),
        sparse_grad=not args.no_sparse_grad,
    )
    loss_function = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss()

    context = get_context(args)
    embedding.initialize(ctx=context)
    embedding_out.initialize(ctx=context)
    if not args.no_hybridize:
        embedding.hybridize(static_alloc=not args.no_static_alloc)
        embedding_out.hybridize(static_alloc=not args.no_static_alloc)

    trainer_emb_in = trainer.get_embedding_in_trainer(
        args, embedding.embedding.collect_params(), len(vocab))
    trainer_emb_out = trainer.get_embedding_out_trainer(
        args, embedding_out.collect_params())

    if args.subword_network.lower() == 'fasttext':
        trainer_subwords = trainer.get_subword_trainer(
            args, embedding.subword_embedding.collect_params(),
            len(subword_function))
    elif args.subword_network.lower() == 'highwaycnn':
        trainer_subwords = trainer.get_subword_trainer(
            args,
            list(embedding.cnn.collect_params().values()) +
            list(embedding.character_embedding.collect_params().values()),
            len(subword_function))

    if args.no_lazy_update:
        trainer_emb_in._optimizer.lazy_update = False
        trainer_emb_out._optimizer.lazy_update = False
        trainer_subwords._optimizer.lazy_update = False

    if args.subword_network.lower() in ['fasttext', 'highwaycnn']:
        minlength = 1
        if hasattr(embedding, 'subwordminlength'):
            minlength = embedding.subwordminlength

    def skipgram_batch(data):
        """Create a batch for Skipgram training objective."""
        centers, word_context, word_context_mask = data
        assert len(centers.shape) == 2
        negatives_shape = (len(word_context), 2 * args.window * args.negative)
        negatives, negatives_mask = remove_accidental_hits(
            negatives_sampler(negatives_shape), word_context,
            word_context_mask)
        context_negatives = mx.nd.concat(word_context, negatives, dim=1)
        masks = mx.nd.concat(word_context_mask, negatives_mask, dim=1)
        labels = mx.nd.concat(word_context_mask, mx.nd.zeros_like(negatives),
                              dim=1)
        if args.subword_network.lower() not in ['fasttext', 'highwaycnn']:
            return (centers.as_in_context(context[0]),
                    context_negatives.as_in_context(context[0]),
                    masks.as_in_context(context[0]),
                    labels.as_in_context(context[0]))
        else:
            unique, inverse_unique_indices = np.unique(centers.asnumpy(),
                                                       return_inverse=True)
            inverse_unique_indices = mx.nd.array(inverse_unique_indices,
                                                 ctx=context[0])
            subwords, subwords_mask = get_subwords_masks(
                unique.astype(int), minlength)

            # Force update of parameters needed for forward pass
            # No-op if parameter was updated during the last iteration.
            # Otherwise equivalent to the parameter being updated with a 0
            # gradient during the last iteration.
            if ('proximal' in args.optimizer.lower()
                    and trainer_emb_in._optimizer.num_update > 0):
                word_fake_grad = mx.nd.sparse.row_sparse_array(
                    (mx.nd.zeros((len(unique), args.emsize)), unique),
                    shape=embedding.embedding.weight.shape)
                word_weight = embedding.embedding.weight
                word_weight.grad()[:] = word_fake_grad
                word_weight.data()._fresh_grad = True
                trainer_emb_in._optimizer._index_update_count[0] -= 1
                trainer_emb_in._optimizer.num_update -= 1
                trainer_emb_in.step(batch_size=1)
            if ('proximal' in args.subword_sparse_optimizer.lower()
                    and trainer_emb_in._optimizer.num_update > 0):
                ngram_unique = np.unique(subwords)
                subword_fake_grad = mx.nd.sparse.row_sparse_array(
                    (mx.nd.zeros(
                        (len(ngram_unique), args.emsize)), ngram_unique),
                    shape=embedding.subword_embedding.embedding.weight.shape)
                subword_weight = embedding.subword_embedding.embedding.weight
                subword_weight.grad()[:] = subword_fake_grad
                subword_weight.data()._fresh_grad = True
                trainer_subwords._optimizer._index_update_count[0] -= 1
                trainer_subwords._optimizer.num_update -= 1
                trainer_subwords.step(batch_size=1)

            return (centers.as_in_context(context[0]),
                    context_negatives.as_in_context(context[0]),
                    masks.as_in_context(context[0]),
                    labels.as_in_context(context[0]),
                    mx.nd.array(subwords, ctx=context[0]),
                    mx.nd.array(subwords_mask, ctx=context[0]),
                    inverse_unique_indices)

    def cbow_batch(data):
        """Create a batch for CBOW training objective."""
        centers, word_context, word_context_mask = data
        assert len(centers.shape) == 2
        negatives_shape = (len(centers), args.negative)
        negatives, negatives_mask = remove_accidental_hits(
            negatives_sampler(negatives_shape), centers)
        center_negatives = mx.nd.concat(centers, negatives, dim=1)
        center_negatives_mask = mx.nd.concat(
            mx.nd.ones_like(centers), negatives_mask, dim=1)
        labels = mx.nd.concat(
            mx.nd.ones_like(centers), mx.nd.zeros_like(negatives), dim=1)
        if args.subword_network.lower() not in ['fasttext', 'highwaycnn']:
            return (word_context.as_in_context(context[0]),
                    word_context_mask.as_in_context(context[0]),
                    center_negatives.as_in_context(context[0]),
                    center_negatives_mask.as_in_context(context[0]),
                    labels.as_in_context(context[0]))
        else:
            unique, inverse_unique_indices = np.unique(word_context.asnumpy(),
                                                       return_inverse=True)
            inverse_unique_indices = mx.nd.array(inverse_unique_indices,
                                                 ctx=context[0])
            subwords, subwords_mask = get_subwords_masks(
                unique.astype(int), minlength)
            return (word_context.as_in_context(context[0]),
                    word_context_mask.as_in_context(context[0]),
                    center_negatives.as_in_context(context[0]),
                    center_negatives_mask.as_in_context(context[0]),
                    labels.as_in_context(context[0]),
                    mx.nd.array(subwords, ctx=context[0]),
                    mx.nd.array(subwords_mask, ctx=context[0]),
                    inverse_unique_indices)

    # Helpers for bucketing
    def skipgram_length_fn(data):
        """Return lengths for bucketing."""
        centers, _, _ = data
        lengths = [
            len(idx_to_subwordidxs[i])
            for i in centers.asnumpy().astype(int).flat
        ]
        return lengths

    def cbow_length_fn(data):
        """Return lengths for bucketing."""
        _, word_context, _ = data
        word_context_np = word_context.asnumpy().astype(int)
        lengths = [
            max(len(idx_to_subwordidxs[i]) for i in one_context)
            for one_context in word_context_np
        ]
        return lengths

    def bucketing_batchify_fn(indices, data):
        """Select elements from data batch based on bucket indices."""
        centers, word_context, word_context_mask = data
        return (centers[indices], word_context[indices],
                word_context_mask[indices])

    length_fn = skipgram_length_fn if args.model.lower() == 'skipgram' \
        else cbow_length_fn

    num_update = 0
    for epoch in range(args.epochs):
        if args.max_words and num_update > args.max_words:
            break

        bucketing_split = 16
        batches = nlp.data.ContextStream(
            stream=data, batch_size=args.batch_size * bucketing_split
            if args.ngram_buckets else args.batch_size,
            p_discard=idx_to_pdiscard, window_size=args.window)
        batches = nlp.data.PrefetchingStream(batches, worker_type='process')
        if args.ngram_buckets:
            # For fastText training, create batches such that subwords used in
            # that batch are of similar length
            batches = nlp.data.BucketingStream(
                batches, bucketing_split, length_fn, bucketing_batchify_fn)

        # Logging variables
        log_wc = 0
        log_start_time = time.time()
        log_avg_loss = 0

        for i, batch in enumerate(batches):
            progress = (epoch * num_tokens + i * args.batch_size) / \
                (args.epochs * num_tokens)

            if args.max_words and num_update > args.max_words:
                break

            if args.model.lower() == 'skipgram':
                if args.ngram_buckets:
                    (center, context_negatives, mask, label, subwords,
                     subwords_mask,
                     inverse_unique_indices) = skipgram_batch(batch)
                    with mx.autograd.record():
                        emb_in = embedding(center, subwords,
                                           subwordsmask=subwords_mask,
                                           words_to_unique_subwords_indices=
                                           inverse_unique_indices)
                        emb_out = embedding_out(context_negatives, mask)
                        pred = mx.nd.batch_dot(emb_in, emb_out.swapaxes(1, 2))
                        loss = loss_function(pred, label)
                else:
                    (center, context_negatives, mask,
                     label) = skipgram_batch(batch)
                    with mx.autograd.record():
                        emb_in = embedding(center)
                        emb_out = embedding_out(context_negatives, mask)
                        pred = mx.nd.batch_dot(emb_in, emb_out.swapaxes(1, 2))
                        loss = loss_function(pred, label)
            elif args.model.lower() == 'cbow':
                if args.ngram_buckets:
                    (word_context, word_context_mask, center_negatives,
                     center_negatives_mask, label, subwords, subwords_mask,
                     inverse_unique_indices) = cbow_batch(batch)
                    with mx.autograd.record():
                        emb_in = embedding(word_context, subwords,
                                           wordsmask=word_context_mask,
                                           subwordsmask=subwords_mask,
                                           words_to_unique_subwords_indices=
                                           inverse_unique_indices)
                        emb_in = emb_in.mean(axis=1, keepdims=True)
                        emb_out = embedding_out(
                            center_negatives, wordsmask=center_negatives_mask)
                        pred = mx.nd.batch_dot(emb_in, emb_out.swapaxes(1, 2))
                        loss = loss_function(pred.squeeze(), label)
                else:
                    (word_context, word_context_mask, center_negatives,
                     center_negatives_mask, label) = cbow_batch(batch)
                    with mx.autograd.record():
                        emb_in = embedding(word_context,
                                           wordsmask=word_context_mask)
                        emb_in = emb_in.mean(axis=1, keepdims=True)
                        emb_out = embedding_out(
                            center_negatives, wordsmask=center_negatives_mask)
                        pred = mx.nd.batch_dot(emb_in, emb_out.swapaxes(1, 2))
                        loss = loss_function(pred.squeeze(), label)
            else:
                logging.error('Unsupported model %s.', args.model)
                sys.exit(1)

            loss.backward()
            num_update += len(label)

            if 'adagrad' not in args.optimizer.lower() \
               or args.adagrad_decay_states:
                trainer_emb_in.set_learning_rate(
                    max(0.0001, args.lr * (1 - progress)))
                trainer_emb_out.set_learning_rate(
                    max(0.0001, args.lr * (1 - progress)))

            if args.subword_network.lower() in ['fasttext', 'highwaycnn'] \
                    and ('adagrad' not in args.subword_sparse_optimizer.lower()
                         or args.adagrad_decay_states):
                trainer_subwords.set_learning_rate(
                    max(0.0001, args.subword_sparse_lr * (1 - progress)))

            if (((i + 1) % args.log_interval == 0) or
                (args.eval_interval and ((i + 1) % args.eval_interval == 0))):
                if 'proximal' in args.optimizer:
                    trainer_emb_in._optimizer.lazy_update = False
                if (args.subword_network.lower() == 'fasttext'
                        and 'proximal' in args.subword_sparse_optimizer):
                    trainer_subwords._optimizer.lazy_update = False
            trainer_emb_in.step(batch_size=1)
            trainer_emb_out.step(batch_size=1)
            if not args.no_lazy_update:
                trainer_emb_in._optimizer.lazy_update = True
            if args.subword_network.lower() in ['fasttext', 'highwaycnn']:
                trainer_subwords.step(batch_size=1)
                if not args.no_lazy_update:
                    trainer_subwords._optimizer.lazy_update = True

            # Logging
            log_wc += loss.shape[0]
            log_avg_loss += loss.mean()
            if (i + 1) % args.log_interval == 0:
                # Forces waiting for computation by computing loss value
                log_avg_loss = log_avg_loss.asscalar() / args.log_interval
                wps = log_wc / (time.time() - log_start_time)
                vector_norm = embedding.embedding.weight.data().norm(axis=1)
                # Due to subsampling, the overall number of batches is an upper bound
                logging.info(
                    '[Epoch {} Batch {}/{}] loss={:.4f}, '
                    'throughput={:.2f}K wps, wc={:.2f}K, '
                    'min_norm={:.2f}, mean_norm={:.2f}, max_norm={:.2f}'.
                    format(epoch, i + 1, num_tokens // args.batch_size,
                           log_avg_loss, wps / 1000, log_wc / 1000,
                           vector_norm.min().asscalar(),
                           vector_norm.mean().asscalar(),
                           vector_norm.max().asscalar()))

                num_zero_word_vectors = mx.nd.sum(
                    vector_norm < 1E-5).asscalar()
                assert len(vocab) == vector_norm.shape[0]

                log_dict = dict(
                    global_step=num_update,
                    epoch=epoch,
                    batch=i + 1,
                    loss=log_avg_loss,
                    wps=wps / 1000,
                    zero_word_vectors=num_zero_word_vectors,
                    nonzero_word_vectors=len(vocab) - num_zero_word_vectors,
                    word_vector_norm_mean=vector_norm.mean().asscalar(),
                    word_vector_norm_min=vector_norm.min().asscalar(),
                    word_vector_norm_max=vector_norm.max().asscalar(),
                )

                if args.subword_network.lower() == 'fasttext':
                    subword_idx_to_vec = embedding.subword_embedding.embedding.weight.data(
                        ctx=context[0]).as_in_context(
                            mx.cpu()).tostype('default')
                    subword_embedding_norm = mx.nd.norm(
                        subword_idx_to_vec, axis=1)
                    num_zero_subword_vectors = mx.nd.sum(
                        subword_embedding_norm < 1E-5).asscalar()
                    assert len(subword_function) == \
                        subword_embedding_norm.shape[0]

                    log_dict = dict(
                        zero_subword_vectors=num_zero_subword_vectors,
                        nonzero_subword_vectors=len(subword_function) -
                        num_zero_subword_vectors,
                        subword_vector_norm_mean=subword_embedding_norm.mean()
                        .asscalar(),
                        subword_vector_norm_min=subword_embedding_norm.min()
                        .asscalar(),
                        subword_vector_norm_max=subword_embedding_norm.max()
                        .asscalar(),
                        **log_dict,
                    )

                log(args, log_dict)

                log_start_time = time.time()
                log_avg_loss = 0
                log_wc = 0

            if args.eval_interval and (i + 1) % args.eval_interval == 0:
                with print_time('mx.nd.waitall()'):
                    mx.nd.waitall()
                with print_time('evaluate'):
                    evaluate(args, embedding, vocab, num_update)

    # Save params
    with print_time('save'):
        save(args, embedding, embedding_out, vocab)

    # Evaluate
    with print_time('mx.nd.waitall()'):
        mx.nd.waitall()
    with print_time('evaluate'):
        evaluate(args, embedding, vocab, num_update,
                 eval_analogy=not args.no_eval_analogy)


def evaluate(args, embedding, vocab, global_step, eval_analogy=False):
    """Evaluation helper"""
    if 'eval_tokens' not in globals():
        global eval_tokens

        eval_tokens_set = evaluation.get_tokens_in_evaluation_datasets(args)
        if not args.no_eval_analogy:
            eval_tokens_set.update(vocab.idx_to_token)

        if args.subword_network.lower() not in ['fasttext', 'highwaycnn']:
            # Word2Vec does not support computing vectors for OOV words
            # TODO replace with __contains__ check
            eval_tokens_set = filter(lambda t: t in vocab, eval_tokens_set)

        eval_tokens = list(eval_tokens_set)

    # Compute their word vectors
    context = get_context(args)
    mx.nd.waitall()

    # Move embedding parameters temporarily to CPU to save GPU memory
    embedding.collect_params().reset_ctx(mx.cpu())

    token_embedding = nlp.embedding.TokenEmbedding(unknown_token=None,
                                                   allow_extend=True)
    token_embedding[eval_tokens] = embedding[eval_tokens]

    # Compute set of vectors with zero word embedding
    zero_word_vectors_words = [
        vocab.idx_to_token[idx] for idx in np.where((
            embedding.embedding.weight.data().norm(axis=1) < 1E-5
        ).asnumpy())[0]
    ]

    evaluation.evaluate_similarity(
        args, token_embedding, context[0],
        zero_word_vectors_set=zero_word_vectors_words, logfile=os.path.join(
            args.logdir, 'similarity.tsv'), global_step=global_step)
    if eval_analogy:
        assert not args.no_eval_analogy
        evaluation.evaluate_analogy(
            args, token_embedding, context[0],
            zero_word_vectors_set=zero_word_vectors_words,
            logfile=os.path.join(args.logdir, 'analogy.tsv'))

    # Move embedding parameters back to GPU
    embedding.collect_params().reset_ctx(context)


def log(args, kwargs):
    logfile = os.path.join(args.logdir, 'log.tsv')

    if 'log_created' not in globals():
        if os.path.exists(logfile):
            logging.error(f'Logfile {logfile} already exists.')
            sys.exit(1)

        global log_created

        log_created = kwargs.keys()
        header = '\t'.join((str(k) for k in kwargs.keys())) + '\n'
        with open(logfile, 'w') as f:
            f.write(header)

    # Log variables shouldn't change during training
    assert log_created == kwargs.keys()

    with open(logfile, 'a') as f:
        f.write('\t'.join((str(v) for v in kwargs.values())) + '\n')


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    args_ = parse_args()

    # Check logdir
    if os.path.exists(args_.logdir):
        newlogdir = tempfile.mkdtemp(dir=args_.logdir)
        logging.warning(f'{args_.logdir} exists. Using {newlogdir}')
        args_.logdir = newlogdir

    os.makedirs(args_.logdir, exist_ok=True)

    train(args_)
