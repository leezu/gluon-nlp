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
import argparse
import itertools
import logging
import gc
import math
import os
import random
import sys
import tempfile
import time
import warnings

import gluonnlp as nlp
from gluonnlp.base import numba_jitclass, numba_prange, numba_types
import mxnet as mx
import numpy as np

import trainer
import evaluation
from candidate_sampler import remove_accidental_hits
from data import WikiDumpStream
from stream import BucketingStream
from utils import get_context, print_time

os.environ['MXNET_GPU_MEM_POOL_TYPE'] = 'Round'

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
    group.add_argument('--batch-size', type=int, default=1024,
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
    group.add_argument(
        '--eval-only', type=str, help='Only evaluate the model '
        'stored at `--eval-only path`')
    group.add_argument('--eval-max-vocab-size', type=int)
    group.add_argument(
        '--alternative-subsampling', action='store_true')

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
    group.add_argument('--max-vocab-size', type=int,
                       help='Limit the number of words considered. '
                       'OOV words will be ignored.')

    # Optimization options
    group.add_argument('--seed', type=int, default=1, help='random seed')
    group.add_argument('--no-zero-init', action='store_true')
    group.add_argument('--no-bucketing', action='store_true')

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
        """Text8 dataset helper."""
        data = nlp.data.Text8(segment='train')
        counter = nlp.data.count_tokens(itertools.chain.from_iterable(data))
        vocab = nlp.Vocab(
            counter,
            unknown_token=None,
            padding_token=None,
            bos_token=None,
            eos_token=None,
            min_freq=5,
            max_size=args.max_vocab_size)
        idx_to_counts = [counter[w] for w in vocab.idx_to_token]
        data = nlp.data.SimpleDataStream([data])
        return data, vocab, idx_to_counts

    def wiki():
        """Wikipedia dump helper."""
        data = WikiDumpStream(
            root=os.path.expanduser(args.wiki_root),
            language=args.wiki_language, date=args.wiki_date)
        vocab = data.vocab
        if args.max_vocab_size:
            for token in vocab.idx_to_token[args.max_vocab_size:]:
                vocab.token_to_idx.pop(token)
            vocab.idx_to_token = vocab.idx_to_token[:args.max_vocab_size]
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

    sum_counts = float(sum(idx_to_counts))
    if not args.alternative_subsampling:
        idx_to_pdiscard = [
            1 - math.sqrt(args.frequent_token_subsampling / (count / sum_counts))
            for count in idx_to_counts
        ]
    else:
        idx_to_pdiscard = [
            1 - math.sqrt(
                args.frequent_token_subsampling / (count / sum_counts)) -
            (args.frequent_token_subsampling / (count / sum_counts))
            for count in idx_to_counts
        ]

    def subsample(shard):
        return [[
            t for t, r in zip(sentence, np.random.uniform(0, 1, size=len(sentence)))
            if r > idx_to_pdiscard[t]
        ] for sentence in shard]

    data = data.transform(subsample)

    return data, negatives_sampler, vocab, sum_counts


def get_subword_functionality(args, vocab):
    with print_time('prepare subwords'):
        if args.subword_network.lower() == 'fasttext':
            subword_function = nlp.vocab.create_subword_function(
                'NGramHashes', ngrams=args.ngrams,
                num_subwords=args.ngram_buckets)
        elif args.subword_network.lower() == 'highwaycnn':
            subword_function = nlp.vocab.create_subword_function(
                'ByteSubwords')
        else:
            raise ValueError('Invalid --subword-network')

        # Store subword indices for all words in vocabulary
        idx_to_subwordidxs = list(subword_function(vocab.idx_to_token))
        subword_lookup = subword_lookup_factory(idx_to_subwordidxs)
        max_subwordidxs_len = max(len(s) for s in idx_to_subwordidxs)
        if max_subwordidxs_len > 500:
            warnings.warn(
                'The word with largest number of subwords '
                'has {} subwords, suggesting there are '
                'some noisy words in your vocabulary. '
                'You should filter out very long words '
                'to avoid memory issues.'.format(max_subwordidxs_len))

    return subword_function, subword_lookup, idx_to_subwordidxs


def subword_lookup_factory(idx_to_subwordidxs):
    """Create a SubwordLookup and initialize idx_to_subwordidxs mapping."""
    subword_lookup = SubwordLookup(len(idx_to_subwordidxs))
    for i, subwords in enumerate(idx_to_subwordidxs):
        subword_lookup.set(i, np.array(subwords, dtype=np.int_))
    return subword_lookup


@numba_jitclass([('idx_to_subwordidxs',
                  numba_types.List(numba_types.int_[::1]))])
class SubwordLookup(object):
    """Just-in-time compiled helper class for fast, padded subword lookup.

    SubwordLookup holds a mapping from token indices to variable length subword
    arrays and allows fast access to padded and masked batches of subwords
    given a list of token indices.

    Parameters
    ----------
    length : int
         Number of tokens for which to hold subword arrays.

    """
    def __init__(self, length):
        self.idx_to_subwordidxs = [
            np.arange(1).astype(np.int_) for _ in range(length)
        ]

    def set(self, i, subwords):
        """Set the subword array of the i-th token."""
        self.idx_to_subwordidxs[i] = subwords

    def get(self, indices, minlength):
        """Get a padded array and mask of subwords for specified indices."""
        subwords = [self.idx_to_subwordidxs[i] for i in indices]
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


def save(args, embedding, embedding_out, vocab, epoch=None):
    """Save parameters to logdir.

    The parameters are first written to a temporary file and only if the saving
    was successful atomically moved to the final location.

    """
    f, path = tempfile.mkstemp(dir=args.logdir)
    os.close(f)

    # save vocab
    with open(path, 'w') as f:
        f.write(vocab.to_json())
    os.replace(path, os.path.join(args.logdir, 'vocab.json'))

    # save list of words with zero word vectors
    context = get_context(args)
    zero_word_vectors_words = sorted([
        vocab.idx_to_token[idx] for idx in np.where((
            embedding.embedding.weight.data(context[0]).norm(axis=1) < 1E-5
        ).asnumpy())[0]
    ])
    with open(path, 'w') as f:
        f.write('\n'.join(zero_word_vectors_words))
    os.replace(
        path,
        os.path.join(
            args.logdir, f'zero_word_vectors_words-{epoch}.txt'
            if epoch is not None else 'zero_word_vectors_words.txt'))

    # write to temporary file; use os.replace
    embedding.save_parameters(os.path.join(
        args.logdir, f'embedding-{epoch}.params'
        if epoch is not None else 'embedding.params'))
    embedding_out.save_parameters(os.path.join(
        args.logdir, f'embedding_out-{epoch}.params'
        if epoch is not None else 'embedding_out.params'))


###############################################################################
# Training code
###############################################################################
def train(args):
    """Training helper."""
    data, negatives_sampler, vocab, num_tokens = \
        get_train_data(args)
    if args.subword_network.lower() == 'fasttext':
        subword_function, subword_lookup, idx_to_subwordidxs = \
            get_subword_functionality(args, vocab)
        embedding = nlp.model.train.FasttextEmbeddingModel(
            token_to_idx=vocab.token_to_idx,
            subword_function=subword_function,
            embedding_size=args.emsize,
            weight_initializer=mx.init.Uniform(scale=1 / args.emsize),
            sparse_grad=not args.no_sparse_grad,
        )
    elif args.subword_network.lower() == 'highwaycnn':
        subword_function, subword_lookup, idx_to_subwordidxs = \
            get_subword_functionality(args, vocab)
        import blocks
        embedding = blocks.HighwayCNNEmbeddingModel(
            token_to_idx=vocab.token_to_idx,
            subword_function=subword_function,
            embedding_size=args.emsize,
            character_embedding_size=15,
            sparse_grad=not args.no_sparse_grad,
        )
    else:
        embedding = nlp.model.train.SimpleEmbeddingModel(
            token_to_idx=vocab.token_to_idx,
            embedding_size=args.emsize,
            weight_initializer=mx.init.Uniform(scale=1 / args.emsize),
            sparse_grad=not args.no_sparse_grad,
        )
    embedding_out = nlp.model.train.SimpleEmbeddingModel(
        token_to_idx=vocab.token_to_idx,
        embedding_size=args.emsize,
        weight_initializer=mx.init.Zero()
        if not args.no_zero_init else mx.init.Uniform(scale=1 / args.emsize),
        sparse_grad=not args.no_sparse_grad,
    )
    loss_function = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss()

    context = get_context(args)
    embedding.initialize(ctx=context)
    embedding_out.initialize(ctx=context)
    if not args.no_hybridize:
        embedding.hybridize(static_alloc=not args.no_static_alloc)
        embedding_out.hybridize(static_alloc=not args.no_static_alloc)

    params_emb_in = list(embedding.embedding.collect_params().values())
    params_emb_out = list(embedding_out.collect_params().values())
    optimizer_emb_in = trainer.get_embedding_in_optimizer(args, len(vocab))
    optimizer_emb_out = trainer.get_embedding_out_optimizer(args)
    trainer_emb_in = mx.gluon.Trainer(params_emb_in, optimizer_emb_in,
                                      kvstore=None)
    trainer_emb_out = mx.gluon.Trainer(params_emb_out, optimizer_emb_out,
                                       kvstore=None)

    if args.subword_network:
        if args.subword_network.lower() == 'fasttext':
            optimizer_subwords = trainer.get_subword_optimizer(
                args, len(subword_function))
            params_subwords = list(
                embedding.subword_embedding.collect_params().values())
        elif args.subword_network.lower() == 'highwaycnn':
            optimizer_subwords = trainer.get_subword_optimizer(
                args, len(subword_function))
            params_subwords = list(embedding.cnn.collect_params().values()) + \
                list(embedding.character_embedding.collect_params().values())
        else:
            print('Unsupported subword network {args.subword_network}'.format(
                args.subword_network))

        trainer_subwords = mx.gluon.Trainer(
            params_subwords, optimizer_subwords, kvstore=None)

    if args.no_lazy_update:
        trainer_emb_in._optimizer.lazy_update = False
        trainer_emb_out._optimizer.lazy_update = False
        trainer_subwords._optimizer.lazy_update = False

    if args.subword_network.lower() in ['fasttext', 'highwaycnn']:
        minlength = 1
        if hasattr(embedding, 'subwordminlength'):
            minlength = embedding.subwordminlength

    def skipgram_batch(data, ctx):
        """Create a batch for Skipgram training objective."""
        centers, word_context, word_context_mask = data
        assert len(centers.shape) == 2
        negatives_shape = (len(word_context), 2 * args.window * args.negative)
        negatives, negatives_mask = remove_accidental_hits(
            negatives_sampler(negatives_shape), word_context)
        context_negatives = mx.nd.concat(word_context, negatives, dim=1)
        masks = mx.nd.concat(word_context_mask, negatives_mask, dim=1)
        labels = mx.nd.concat(word_context_mask, mx.nd.zeros_like(negatives),
                              dim=1)
        if args.subword_network.lower() not in ['fasttext', 'highwaycnn']:
            return (centers.as_in_context(ctx),
                    context_negatives.as_in_context(ctx),
                    masks.as_in_context(ctx),
                    labels.as_in_context(ctx))
        else:
            unique, inverse_unique_indices = np.unique(centers.asnumpy(),
                                                       return_inverse=True)
            inverse_unique_indices = mx.nd.array(inverse_unique_indices,
                                                 ctx=ctx)
            subwords, subwords_mask = subword_lookup.get(
                unique.astype(int), minlength)

            # Force update of parameters needed for forward pass
            # No-op if parameter was updated during the last iteration.
            # Otherwise equivalent to the parameter being updated with a 0
            # gradient during the last iteration.
            if 'proximal' in args.optimizer.lower() \
               and trainer_emb_in._optimizer.num_update > 0 \
               and args.l2 > 0:
                word_fake_grad = mx.nd.sparse.row_sparse_array(
                    (mx.nd.zeros((len(unique), args.emsize)), unique),
                    shape=embedding.embedding.weight.shape,
                    ctx=ctx)
                # trainer_emb_in._optimizer is shared among all updaters
                # trainer_emb_in._updaters[ctx_idx]
                trainer_emb_in._optimizer._index_update_count[0] -= 1
                trainer_emb_in._optimizer.num_update -= 1
                ctx_idx = trainer_emb_in._contexts.index(ctx)
                upd = trainer_emb_in._updaters[ctx_idx]
                assert len(trainer_emb_in._params) == 1
                for i, param in enumerate(trainer_emb_in._params):
                    arr = param.list_data()[ctx_idx]
                    upd(i, word_fake_grad, arr)
            if 'proximal' in args.subword_sparse_optimizer.lower() \
                    and trainer_emb_in._optimizer.num_update > 0 \
                    and args.subword_sparse_l2 > 0:
                ngram_unique = np.unique(subwords)
                subword_fake_grad = mx.nd.sparse.row_sparse_array(
                    (mx.nd.zeros((len(ngram_unique), args.emsize)),
                     ngram_unique),
                    shape=embedding.subword_embedding.embedding.weight.shape,
                    ctx=ctx)
                trainer_subwords._optimizer._index_update_count[0] -= 1
                trainer_subwords._optimizer.num_update -= 1
                ctx_idx = trainer_subwords._contexts.index(ctx)
                upd = trainer_subwords._updaters[ctx_idx]
                assert len(trainer_subwords._params) == 1
                for i, param in enumerate(trainer_subwords._params):
                    arr = param.list_data()[ctx_idx]
                    upd(i, subword_fake_grad, arr)

            return (centers.as_in_context(ctx),
                    context_negatives.as_in_context(ctx),
                    masks.as_in_context(ctx),
                    labels.as_in_context(ctx),
                    mx.nd.array(subwords, ctx=ctx),
                    mx.nd.array(subwords_mask, ctx=ctx),
                    inverse_unique_indices)

    def cbow_batch(data, ctx):
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
            return (word_context.as_in_context(ctx),
                    word_context_mask.as_in_context(ctx),
                    center_negatives.as_in_context(ctx),
                    center_negatives_mask.as_in_context(ctx),
                    labels.as_in_context(ctx))
        else:
            unique, inverse_unique_indices = np.unique(word_context.asnumpy(),
                                                       return_inverse=True)
            inverse_unique_indices = mx.nd.array(inverse_unique_indices,
                                                 ctx=ctx)
            subwords, subwords_mask = subword_lookup.get(
                unique.astype(int), minlength)
            return (word_context.as_in_context(ctx),
                    word_context_mask.as_in_context(ctx),
                    center_negatives.as_in_context(ctx),
                    center_negatives_mask.as_in_context(ctx),
                    labels.as_in_context(ctx),
                    mx.nd.array(subwords, ctx=ctx),
                    mx.nd.array(subwords_mask, ctx=ctx),
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


    bucketing_split = 16
    batchify = nlp.data.batchify.EmbeddingCenterContextBatchify(
        batch_size=args.batch_size * bucketing_split
        if args.ngram_buckets else args.batch_size,
        window_size=args.window)
    data = data.transform(batchify)

    num_update = 0
    for epoch in range(args.epochs):
        if args.max_words and num_update > args.max_words:
            break
        # Logging variables
        log_wc = 0
        log_start_time = time.time()
        log_avg_loss = 0

        batches = itertools.chain.from_iterable(data)

        if args.ngram_buckets and not args.no_bucketing:
            # For fastText training, create batches such that subwords used in
            # that batch are of similar length
            batches = BucketingStream(batches, bucketing_split, length_fn,
                                      bucketing_batchify_fn)

        for i, batch in enumerate(batches):
            ctx = context[i % len(context)]
            progress = (epoch * num_tokens + i * args.batch_size) / \
                (args.epochs * num_tokens)

            if args.max_words and num_update > args.max_words:
                break

            if args.model.lower() == 'skipgram':
                if args.ngram_buckets:
                    (center, context_negatives, mask, label, subwords,
                     subwords_mask,
                     inverse_unique_indices) = skipgram_batch(batch, ctx)
                    with mx.autograd.record():
                        emb_in = embedding(center, subwords,
                                           subwordsmask=subwords_mask,
                                           words_to_unique_subwords_indices=
                                           inverse_unique_indices)
                        emb_out = embedding_out(context_negatives, mask)
                        pred = mx.nd.batch_dot(emb_in, emb_out.swapaxes(1, 2))
                        loss = (loss_function(pred, label, mask) *
                                mask.shape[1] / mask.sum(axis=1))
                else:
                    (center, context_negatives, mask,
                     label) = skipgram_batch(batch, ctx)
                    with mx.autograd.record():
                        emb_in = embedding(center)
                        emb_out = embedding_out(context_negatives, mask)
                        pred = mx.nd.batch_dot(emb_in, emb_out.swapaxes(1, 2))
                        loss = (loss_function(pred, label, mask) *
                                mask.shape[1] / mask.sum(axis=1))
            elif args.model.lower() == 'cbow':
                if args.ngram_buckets:
                    (word_context, word_context_mask, center_negatives,
                     center_negatives_mask, label, subwords, subwords_mask,
                     inverse_unique_indices) = cbow_batch(batch, ctx)
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
                        loss = (loss_function(pred.squeeze(), label,
                                              center_negatives_mask) *
                                center_negatives_mask.shape[1] /
                                center_negatives_mask.sum(axis=1))
                else:
                    (word_context, word_context_mask, center_negatives,
                     center_negatives_mask, label) = cbow_batch(batch, ctx)
                    with mx.autograd.record():
                        emb_in = embedding(word_context,
                                           wordsmask=word_context_mask)
                        emb_in = emb_in.mean(axis=1, keepdims=True)
                        emb_out = embedding_out(
                            center_negatives, wordsmask=center_negatives_mask)
                        pred = mx.nd.batch_dot(emb_in, emb_out.swapaxes(1, 2))
                        loss = (loss_function(pred.squeeze(), label,
                                              center_negatives_mask) *
                                center_negatives_mask.shape[1] /
                                center_negatives_mask.sum(axis=1))
            else:
                logging.error('Unsupported model %s.', args.model)
                sys.exit(1)

            loss.backward()
            num_update += len(label)

            if 'adagrad' not in args.optimizer.lower() \
               or args.adagrad_decay_states:
                if args.lr_schedule.lower() == 'linear':
                    trainer_emb_in.set_learning_rate(
                        max(0.0001, args.lr * (1 - progress)))
                    trainer_emb_out.set_learning_rate(
                        max(0.0001, args.lr * (1 - progress)))
                    if args.subword_network.lower() in [
                            'fasttext', 'highwaycnn'
                    ]:
                        trainer_subwords.set_learning_rate(
                            max(0.0001,
                                args.subword_sparse_lr * (1 - progress)))
                elif args.lr_schedule.lower() == 'step':
                    decay = args.lr_schedule_step_drop**math.floor(
                        epoch / args.lr_schedule_step_size)
                    trainer_emb_in.set_learning_rate(args.lr * decay)
                    trainer_emb_out.set_learning_rate(args.lr * decay)
                    if args.subword_network.lower() in [
                            'fasttext', 'highwaycnn'
                    ]:
                        trainer_subwords.set_learning_rate(
                            args.subword_sparse_lr * decay)
                else:
                    raise RuntimeError('Invalid learning rate schedule.')

            if (((i + 1) % args.log_interval == 0) or
                (args.eval_interval and ((i + 1) % args.eval_interval == 0))):
                if 'proximal' in args.optimizer and not args.adagrad_groupwise_lr:  # TODO cast grad to dense and add dense support to optimizer as eager update
                    trainer_emb_in._optimizer.lazy_update = False
                    if args.subword_network.lower() == 'fasttext':
                        trainer_subwords._optimizer.lazy_update = False
            if len(context) == 1 or (i + 1) % len(context) == 0:
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
            log_avg_loss += loss.mean().as_in_context(context[0])
            if (i + 1) % args.log_interval == 0:
                # Forces waiting for computation by computing loss value
                log_avg_loss = log_avg_loss.asscalar() / args.log_interval
                wps = log_wc / (time.time() - log_start_time)
                vector_norm = embedding.embedding.weight.data(context[0]).norm(axis=1)
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
                    subword_embedding_norm = embedding.subword_embedding.embedding.weight.data(
                        ctx=context[0]).norm(axis=1)
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
            save(args, embedding, embedding_out, vocab, epoch)

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
            if not args.eval_max_vocab_size:
                eval_tokens_set.update(vocab.idx_to_token)
            else:
                eval_tokens_set.update(vocab.idx_to_token[:args.eval_max_vocab_size])

        if args.subword_network.lower() not in ['fasttext', 'highwaycnn']:
            # Word2Vec does not support computing vectors for OOV words
            # TODO replace with __contains__ check
            eval_tokens_set = filter(lambda t: t in vocab, eval_tokens_set)

        eval_tokens = list(eval_tokens_set)

    # Compute their word vectors
    context = get_context(args)
    mx.nd.waitall()

    # Move embedding parameters temporarily to CPU to save GPU memory
    gc.collect()  # Release CPU memory
    embedding.collect_params().reset_ctx(mx.cpu())

    token_embedding = nlp.embedding.TokenEmbedding(unknown_token=None,
                                                   allow_extend=True)
    with print_time('compute vectors from subwords '
                    'for {} words.'.format(len(eval_tokens))):
        token_embedding[eval_tokens] = embedding[eval_tokens]

    # Compute set of vectors with zero word embedding
    context = get_context(args)
    zero_word_vectors_words = [
        vocab.idx_to_token[idx] for idx in np.where((
            embedding.embedding.weight.data(mx.cpu()).norm(axis=1) < 1E-5
        ).asnumpy())[0]
    ]

    known_tokens = vocab.idx_to_token
    if args.eval_max_vocab_size:
        known_tokens = known_tokens[:args.eval_max_vocab_size]
    known_tokens = set(known_tokens)
    evaluation.evaluate_similarity(
        args, token_embedding, context[0], known_tokens,
        zero_word_vectors_set=zero_word_vectors_words, logfile=os.path.join(
            args.logdir, 'similarity.tsv'), global_step=global_step)
    if eval_analogy:
        assert not args.no_eval_analogy
        evaluation.evaluate_analogy(
            args, token_embedding, context[0], known_tokens,
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


def load_and_evaluate(args):
    """Training helper."""
    with open(os.path.join(args.eval_only, 'vocab.json'), 'r') as f:
        with print_time('load vocab from file'):
            vocab = nlp.Vocab.from_json(f.read())

    with print_time('initialize model'):
        if args.subword_network.lower() == 'fasttext':
            subword_function, subword_lookup, idx_to_subwordidxs = \
                get_subword_functionality(args, vocab)
            embedding = nlp.model.train.FasttextEmbeddingModel(
                token_to_idx=vocab.token_to_idx,
                subword_function=subword_function,
                embedding_size=args.emsize,
                weight_initializer=mx.init.Uniform(scale=1 / args.emsize),
                sparse_grad=not args.no_sparse_grad,
               )
        elif args.subword_network.lower() == 'highwaycnn':
            subword_function, subword_lookup, idx_to_subwordidxs = \
                get_subword_functionality(args, vocab)
            import blocks
            embedding = blocks.HighwayCNNEmbeddingModel(
                token_to_idx=vocab.token_to_idx,
                subword_function=subword_function,
                embedding_size=args.emsize,
                character_embedding_size=15,
                sparse_grad=not args.no_sparse_grad,
               )
        else:
            embedding = nlp.model.train.SimpleEmbeddingModel(
                token_to_idx=vocab.token_to_idx,
                embedding_size=args.emsize,
                weight_initializer=mx.init.Uniform(scale=1 / args.emsize),
                sparse_grad=not args.no_sparse_grad,
               )
        embedding_out = nlp.model.train.SimpleEmbeddingModel(
            token_to_idx=vocab.token_to_idx,
            embedding_size=args.emsize,
            weight_initializer=mx.init.Zero()
            if not args.no_zero_init else mx.init.Uniform(scale=1 / args.emsize),
            sparse_grad=not args.no_sparse_grad,
           )

    context = get_context(args)
    embedding.initialize(ctx=context)
    embedding_out.initialize(ctx=context)
    if not args.no_hybridize:
        embedding.hybridize(static_alloc=not args.no_static_alloc)
        embedding_out.hybridize(static_alloc=not args.no_static_alloc)

    with print_time('load parameters from file'):
        embedding.collect_params().load(
            os.path.join(args.eval_only, 'embedding.params'))
        embedding_out.collect_params().load(
            os.path.join(args.eval_only, 'embedding_out.params'))

    with print_time('evaluate'):
        evaluate(args, embedding, vocab, 0,
                 eval_analogy=not args.no_eval_analogy)


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    args_ = parse_args()

    # Check logdir
    if os.path.exists(args_.logdir) and \
       set(os.listdir(args_.logdir)) - set(("stderr.log", "stdout.log")):
        newlogdir = tempfile.mkdtemp(dir=args_.logdir)
        logging.warning(f'{args_.logdir} exists and contains '
                        f'more than stderr/stdout.log. Using {newlogdir}')
        args_.logdir = newlogdir

    os.makedirs(args_.logdir, exist_ok=True)

    if not args_.eval_only:
        train(args_)
    else:
        load_and_evaluate(args_)
