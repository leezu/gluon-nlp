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
"""Global SkipGram embedding model
==================================

"""
# * Imports
import argparse
import functools
import logging
import os
import random
import sys
import tempfile
import time

import mxnet as mx
import numpy as np

import evaluation
import gluonnlp as nlp
from model import SG
from utils import get_context, print_time
from data import skipgram_lookup, ShuffledBatchedStream, ArrayDataset


# * Utils
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='GloVe with GluonNLP',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Data options
    group = parser.add_argument_group('Data arguments')
    group.add_argument(
        'cooccurrences', type=str,
        help='Path to cooccurrences.npz containing a sparse (COO) '
        'representation of the co-occurrence matrix in numpy archive format. '
        'Output of ./cooccur')
    group.add_argument('vocab', type=str,
                       help='Vocabulary indices. Output of vocab_count tool.')

    # Computation options
    group = parser.add_argument_group('Computation arguments')
    group.add_argument('--batch-size', type=int, default=4096,
                       help='Batch size for training.')
    group.add_argument('--epochs', type=int, default=5, help='Epoch limit')
    group.add_argument(
        '--gpu', type=int, nargs='+',
        help='Number (index) of GPU to run on, e.g. 0. '
        'If not specified, uses CPU.')
    group.add_argument('--no-prefetch-batch', action='store_true',
                       help='Disable multi-threaded nogil batch prefetching.')
    group.add_argument('--no-hybridize', action='store_true',
                       help='Disable hybridization of gluon HybridBlocks.')

    # Model
    group = parser.add_argument_group('Model arguments')
    group.add_argument('--emsize', type=int, default=300,
                       help='Size of embedding vectors.')
    group.add_argument('--ngrams', type=int, nargs='+', default=[3, 4, 5, 6])
    group.add_argument(
        '--ngram-buckets', type=int, default=2000000,
        help='Size of word_context set of the ngram hash function. '
        'Set this to 0 for Word2Vec style training.')
    group.add_argument(
        '--negative', type=int, default=5, help='Number of negative samples '
        'per source-context word pair.')
    group.add_argument(
        '--counts-min', type=int, default=2,
        help='Remove all co-occurrences with less than --counts-min count.')
    group.add_argument(
        '--counts-max', type=int, default=5,
        help='Set all counts greater --counts-max to --counts-max')

    # Optimization options
    group = parser.add_argument_group('Optimization arguments')
    group.add_argument('--adagrad-eps', type=float, default=1,
                       help='Initial AdaGrad state value.')
    group.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    group.add_argument('--seed', type=int, default=1, help='Random seed')

    # Logging
    group = parser.add_argument_group('Logging arguments')
    group.add_argument('--logdir', type=str, default='logs',
                       help='Directory to store logs.')
    group.add_argument('--log-interval', type=int, default=100)
    group.add_argument(
        '--eval-interval', type=int,
        help='Evaluate every --eval-interval iterations '
        'in addition to at the end of every epoch.')
    group.add_argument('--no-eval-analogy', action='store_true',
                       help='Don\'t evaluate on the analogy task.')

    # Evaluation options
    evaluation.add_parameters(parser)

    args = parser.parse_args()
    evaluation.validate_args(args)

    random.seed(args.seed)
    mx.random.seed(args.seed)
    np.random.seed(args.seed)

    return args


def get_train_data(args):
    """Helper function to get training data."""
    counter = dict()
    with open(args.vocab, 'r', encoding='utf-8') as f:
        for line in f:
            token, count = line.split('\t')
            counter[token] = int(count)
    vocab = nlp.Vocab(counter, unknown_token=None, padding_token=None,
                      bos_token=None, eos_token=None, min_freq=1)
    idx_to_counts = [counter[w] for w in vocab.idx_to_token]

    # Prepare subwords
    if args.ngram_buckets:
        with print_time('prepare subwords'):
            subword_function = nlp.vocab.create_subword_function(
                'NGramHashes', ngrams=args.ngrams,
                num_subwords=args.ngram_buckets)

            # Store subword indices for all words in vocabulary
            idx_to_subwordidxs = list(subword_function(vocab.idx_to_token))
            subwordidxs = np.concatenate(idx_to_subwordidxs)
            subwordidxsptr = np.cumsum([
                len(subwordidxs) for subwordidxs in idx_to_subwordidxs])
            subwordidxsptr = np.concatenate([
                np.zeros(1, dtype=np.int64), subwordidxsptr])
            subword_lookup = functools.partial(
                skipgram_lookup, subwordidxs=subwordidxs,
                subwordidxsptr=subwordidxsptr, offset=len(vocab))
    else:
        subword_function = None

    def sg_fasttext_batch(centers, contexts, counts):
        """Create a batch for SG training objective with subwords."""
        data, row, col = subword_lookup(centers.asnumpy())
        centers_csr = mx.nd.sparse.csr_matrix(
            (data, (row, col)), dtype=np.float32,
            shape=(len(centers), len(vocab) + args.ngram_buckets))
        return centers_csr, contexts, centers, counts

    def sg_batch(centers, contexts, counts):
        """Create a batch for SG training objective."""
        indptr = mx.nd.arange(len(centers) + 1)
        centers_csr = mx.nd.sparse.csr_matrix(
            (mx.nd.ones(centers.shape), centers, indptr), dtype=np.float32,
            shape=(len(centers), len(vocab)))
        return centers_csr, contexts, centers, counts

    batchify_fn = sg_batch if not args.ngram_buckets else sg_fasttext_batch

    cooccur = np.load(args.cooccurrences)
    row, col, counts = cooccur['row'], cooccur['col'], cooccur['data']
    valid_count = counts > args.counts_min
    row, col, counts = row[valid_count], col[valid_count], counts[valid_count]
    np.sqrt(counts, out=counts)
    counts[counts > args.counts_max] = args.counts_max

    index_dtype = 'int32'
    if row.max() >= np.iinfo(np.int32).max:
        index_dtype = 'int64'
        # MXNet has no support for uint32, so we must fall back to int64
        logging.info('More words than could be counted using int32. '
                     'Using int64 to represent word indices.')
    row = mx.nd.array(row, dtype=index_dtype)
    col = mx.nd.array(col, dtype=index_dtype)
    counts = mx.nd.array(counts, dtype=np.float32)

    # row is always used as 'source' and col as 'context' word. Therefore
    # duplicate the entries.
    assert row.shape == col.shape
    row = mx.nd.concatenate([row, col])
    col = mx.nd.concatenate([col, row[:len(row) // 2]])
    counts = mx.nd.concatenate([counts, counts])

    data = ArrayDataset(row, col, counts)
    data = ShuffledBatchedStream(data, args.batch_size)

    return (vocab, data, idx_to_counts, index_dtype, subword_function,
            batchify_fn)


# * Training code
def train(args):
    """Training helper."""
    (vocab, data, idx_to_counts, index_dtype, subword_function,
     batchify_fn) = get_train_data(args)
    embedding = SG(token_to_idx=vocab.token_to_idx, output_dim=args.emsize,
                   batch_size=args.batch_size, num_negatives=args.negative,
                   negatives_weights=mx.nd.array(idx_to_counts),
                   subword_function=subword_function, dtype=np.float32,
                   index_dtype=index_dtype)
    context = get_context(args)
    embedding.initialize(ctx=context)
    if not args.no_hybridize:
        embedding.hybridize(static_alloc=True, static_shape=True)

    optimizer_kwargs = dict(learning_rate=args.lr, eps=args.adagrad_eps)
    params = list(embedding.collect_params().values())
    trainer = mx.gluon.Trainer(params, 'groupadagrad', optimizer_kwargs)

    num_batches = len(data)
    try:
        if args.no_prefetch_batch:
            batches = data.transform(batchify_fn)
        else:
            from executors import LazyThreadPoolExecutor
            num_cpu = len(os.sched_getaffinity(0))
            ex = LazyThreadPoolExecutor(num_cpu)
    except (ImportError, SyntaxError, AttributeError):
        # Py2 - no async prefetching is supported
        logging.warning(
            'Asynchronous batch prefetching is not supported on Python 2. '
            'Consider upgrading to Python 3 for improved performance.')
        batches = data.transform(batchify_fn)

    for epoch in range(args.epochs):
        try:
            batches = ex.map(batchify_fn, data)
        except NameError:  # Py 2 or prefetching disabled
            pass

        # Logging variables
        log_wc = 0
        log_start_time = time.time()
        log_avg_loss = 0

        for i, batch in enumerate(batches):
            ctx = context[i % len(context)]
            batch = [array.as_in_context(ctx) for array in batch]
            with mx.autograd.record():
                loss = embedding(*batch)
            loss.backward()

            if len(context) == 1 or (i + 1) % len(context) == 0:
                trainer.step(batch_size=1)

            # Logging
            log_wc += loss.shape[0]
            log_avg_loss += loss.mean().as_in_context(context[0])
            if (i + 1) % args.log_interval == 0:
                # Forces waiting for computation by computing loss value
                log_avg_loss = log_avg_loss.asscalar() / args.log_interval
                wps = log_wc / (time.time() - log_start_time)
                logging.info('[Epoch {} Batch {}/{}] loss={:.4f}, '
                             'throughput={:.2f}K wps, wc={:.2f}K'.format(
                                 epoch, i + 1, num_batches, log_avg_loss,
                                 wps / 1000, log_wc / 1000))
                log_start_time = time.time()
                log_avg_loss = 0
                log_wc = 0

                log_dict = dict(
                    global_step=epoch * num_batches * args.batch_size +
                    i * args.batch_size,
                    epoch=epoch,
                    batch=i + 1,
                    loss=log_avg_loss,
                    wps=wps / 1000,
                )

                log(args, log_dict)

            if args.eval_interval and (i + 1) % args.eval_interval == 0:
                with print_time('mx.nd.waitall()'):
                    mx.nd.waitall()
                with print_time('evaluate'):
                    evaluate(args, embedding, vocab, i + num_batches * epoch)

    # Evaluate
    with print_time('mx.nd.waitall()'):
        mx.nd.waitall()
    with print_time('evaluate'):
        evaluate(args, embedding, vocab, num_batches * args.epochs,
                 eval_analogy=not args.no_eval_analogy)

    # Save params
    with print_time('save parameters'):
        embedding.save_parameters(
            os.path.join(args.logdir, 'embedding.params'))


# * Evaluation
def evaluate(args, embedding, vocab, global_step, eval_analogy=False):
    """Evaluation helper"""
    if 'eval_tokens' not in globals():
        global eval_tokens

        eval_tokens_set = evaluation.get_tokens_in_evaluation_datasets(args)
        if not args.no_eval_analogy:
            eval_tokens_set.update(vocab.idx_to_token)

        if not args.ngram_buckets:
            # Word2Vec does not support computing vectors for OOV words
            eval_tokens_set = filter(lambda t: t in vocab, eval_tokens_set)

        eval_tokens = list(eval_tokens_set)

    if not os.path.isdir(args.logdir):
        os.makedirs(args.logdir)

    # Compute their word vectors
    context = get_context(args)
    mx.nd.waitall()

    token_embedding = nlp.embedding.TokenEmbedding(unknown_token=None,
                                                   allow_extend=True)
    token_embedding[eval_tokens] = embedding[eval_tokens]

    results = evaluation.evaluate_similarity(
        args, token_embedding, context[0], logfile=os.path.join(
            args.logdir, 'similarity.tsv'), global_step=global_step)
    if eval_analogy:
        assert not args.no_eval_analogy
        results += evaluation.evaluate_analogy(
            args, token_embedding, context[0], logfile=os.path.join(
                args.logdir, 'analogy.tsv'))

    return results


# * Logging
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
        # TODO only ordered on Py3.6+
        f.write('\t'.join((str(v) for v in kwargs.values())) + '\n')


# * Main
if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    args_ = parse_args()

    if os.path.exists(args_.logdir):
        newlogdir = tempfile.mkdtemp(dir=args_.logdir)
        logging.warning(f'{args_.logdir} exists. Using {newlogdir}')
        args_.logdir = newlogdir
    os.makedirs(args_.logdir, exist_ok=True)

    train(args_)
