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
"""Word Embeddings Training
===========================

This example shows how to train word embeddings.

"""

import argparse
import logging
import math
import multiprocessing as mp
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

import mxnet as mx
import numpy as np
from mxboard import SummaryWriter
from mxnet import gluon

import data
import evaluation
import fasttext
import gluonnlp as nlp
import subword
import utils

try:
    import tqdm
except ImportError:
    logging.warning('tqdm not installed. '
                    ' Install via `pip install tqdm` for better usability.')
    tqdm = None


def get_args():
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        description='Word embedding training with Gluon.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model arguments
    group = parser.add_argument_group('Model arguments')
    group.add_argument('--no-token-embedding', action='store_true',
                       help='Don\'t use any token embedding. '
                       'Only use subword units.')
    group.add_argument(
        '--subword-network', type=str, default='SubwordCNN',
        help=('Network architecture to infer subword level embeddings. ' +
              str(subword.list_subwordnetworks()) +
              ' , fasttext or empty to disable'))
    group.add_argument('--objective', type=str, default='skipgram',
                       help='Word embedding training objective.')
    group.add_argument('--emsize', type=int, default=300,
                       help='Size of word embeddings')
    group.add_argument(
        '--normalized-initialization', action='store_true',
        help='Normalize uniform initialization range by embedding size.')
    # Evaluation arguments
    group = parser.add_argument_group('Evaluation arguments')
    group.add_argument('--eval-interval', type=int, default=100,
                       help='evaluation interval')
    ## Datasets
    group.add_argument('--train-dataset', type=str, default='Text8',
                       help='Training corpus. [\'Text8\', \'Test\']')
    group.add_argument(
        '--similarity-datasets', type=str,
        default=nlp.data.word_embedding_evaluation.word_similarity_datasets,
        nargs='*',
        help='Word similarity datasets to use for intrinsic evaluation.')
    group.add_argument(
        '--similarity-functions', type=str,
        default=nlp.embedding.evaluation.list_evaluation_functions(
            'similarity'), nargs='+',
        help='Word similarity functions to use for intrinsic evaluation.')

    # Computation options
    group = parser.add_argument_group('Computation arguments')
    group.add_argument('--batch-size', type=int, default=1024,
                       help='Batch size for training.')
    group.add_argument('--sparsity-lambda', type=float, default=0.01,
                       help='Initial learning rate')
    group.add_argument('--lr', type=float, default=0.1,
                       help='Initial learning rate')
    group.add_argument('--epochs', type=int, default=5, help='Epoch limit')
    group.add_argument('--gpu', type=int, nargs='+',
                       help=('Number (index) of GPU to run on, e.g. 0. '
                             'If not specified, uses CPU.'))
    group.add_argument('--dont-hybridize', action='store_true',
                       help='Disable hybridization of gluon HybridBlocks.')
    group.add_argument('--normalize-gradient', type=str, default='count',
                       help='Normalize the word embedding gradient row-wise. '
                       'Supported are [None, count, L2].')
    group.add_argument(
        '--force-py-op-normalize-gradient', action='store_true',
        help='Always use Python sparse L2 normalization operator.')

    # Logging options
    group = parser.add_argument_group('Logging arguments')
    group.add_argument('--logdir', type=str, default=None,
                       help='Directory to store logs in.'
                       'Tensorboard compatible logs are stored there. '
                       'Defaults to a random directory in ./logs')

    args = parser.parse_args()

    return args


###############################################################################
# Parse arguments
###############################################################################
def validate_args(args):
    """Validate provided arguments and act on --help."""
    # Check correctness of similarity dataset names

    context = utils.get_context(args)
    assert args.batch_size % len(context) == 0, \
        "Total batch size must be multiple of the number of devices"

    for dataset_name in args.similarity_datasets:
        if dataset_name.lower() not in map(
                str.lower,
                nlp.data.word_embedding_evaluation.word_similarity_datasets):
            print('{} is not a supported dataset.'.format(dataset_name))
            sys.exit(1)

    if args.no_token_embedding and not args.subword_network:
        raise RuntimeError('At least one of token and subword level embedding '
                           'has to be used')


def setup_logging(args):
    """Set up the logging directory."""

    if not args.logdir:
        args.logdir = tempfile.mkdtemp(dir='./logs')

    logging.info('Logging to {}'.format(args.logdir))


class SubwordEmbeddings(gluon.Block):
    def __init__(self, embedding_dim, subword_network, **kwargs):
        super().__init__(**kwargs)

        self.embedding_dim = embedding_dim

        with self.name_scope():
            if 'rnn' in subword_network.lower():
                self.subword = subword.create(
                    name=subword_network, mode='lstm', embed_size=32,
                    hidden_size=64, output_size=embedding_dim)
            else:
                self.subword = subword.create(name=subword_network,
                                              embed_size=32,
                                              output_size=embedding_dim)

    def forward(self, token_bytes, mask):
        return self.subword(token_bytes, mask)


###############################################################################
# Build the model
###############################################################################
def get_model(args, train_dataset):
    assert not (args.no_token_embedding and not args.subword_network)
    num_tokens = train_dataset.num_tokens
    context = utils.get_context(args)
    embeddings_context = [context[0]]
    embedding_out = gluon.nn.SparseEmbedding(num_tokens, args.emsize)
    if args.normalized_initialization:
        embedding_out.initialize(
            mx.init.Uniform(scale=1 / args.emsize), ctx=embeddings_context)
    else:
        embedding_out.initialize(mx.init.Uniform(), ctx=embeddings_context)
    if not args.dont_hybridize:
        embedding_out.hybridize()

    if not args.no_token_embedding:
        embedding_in = gluon.nn.SparseEmbedding(num_tokens, args.emsize)
        if args.normalized_initialization:
            embedding_in.initialize(
                mx.init.Uniform(scale=1 / args.emsize), ctx=embeddings_context)
        else:
            embedding_in.initialize(mx.init.Uniform(), ctx=embeddings_context)

        if not args.dont_hybridize:
            embedding_in.hybridize()
    else:
        embedding_in = None

    if args.subword_network:
        subword_net = SubwordEmbeddings(embedding_dim=args.emsize,
                                        subword_network=args.subword_network)
        subword_net.initialize(mx.init.Xavier(), ctx=context)

        if not args.dont_hybridize:
            subword_net.hybridize()
    else:
        subword_net = None

    loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()

    return embedding_in, embedding_out, subword_net, loss


###############################################################################
# Training code
###############################################################################
def train(args):
    train_dataset, vocab, subword_vocab = data.get_train_data(args)
    embedding_in, embedding_out, subword_net, loss_function = get_model(
        args, train_dataset)
    context = utils.get_context(args)

    dense_params = list(subword_net.collect_params().values())

    dense_trainer = gluon.Trainer(dense_params, 'sgd', {
        'learning_rate': args.lr,
        'momentum': 0.1
    })

    # Auxilary states for group lasso objective
    last_update_buffer = mx.nd.zeros((train_dataset.num_tokens, ),
                                     ctx=context[0])
    current_update = 1

    # Logging writer
    sw = SummaryWriter(logdir=args.logdir)

    indices = np.arange(len(train_dataset))
    for epoch in range(args.epochs):
        np.random.shuffle(indices)
        batches = [
            indices[i:i + args.batch_size]
            for i in range(0, len(indices), args.batch_size)
        ]

        if tqdm is not None:
            t = tqdm.trange(len(batches), smoothing=1)
        else:
            t = range(len(batches))

        num_workers = math.ceil(mp.cpu_count() * 0.8)
        executor = ThreadPoolExecutor(max_workers=num_workers)
        for i, batch in zip(t, executor.map(train_dataset.__getitem__,
                                            batches)):
            (source, target, label, unique_sources_indices,
             unique_sources_counts, unique_sources_subwordsequences,
             source_subword, unique_sources_subwordsequences_mask,
             unique_targets_indices, unique_targets_counts) = batch

            mx.nd.waitall()

            # Load data for training embedding matrix to context[0]
            source = mx.nd.array(source, ctx=context[0])
            target = mx.nd.array(target, ctx=context[0])
            label = mx.nd.array(label, ctx=context[0])

            # Load indices for looking up subword embedding to context[0]
            source_subword = mx.nd.array(source_subword, ctx=context[0])

            # Split and load subword info to all GPUs for accelerated computation
            assert unique_sources_subwordsequences.shape == unique_sources_subwordsequences_mask.shape
            unique_token_subwordsequences = gluon.utils.split_and_load(
                unique_sources_subwordsequences, context, batch_axis=1,
                even_split=False)
            unique_sources_subwordsequences_mask = gluon.utils.split_and_load(
                unique_sources_subwordsequences_mask, context, batch_axis=1,
                even_split=False)

            with mx.autograd.record():
                if subword_net is not None:
                    # Compute subword embeddings from subword info (byte sequences)
                    subword_embedding_weights = []
                    for subwordsequences_ctx, mask_ctx in zip(
                            unique_token_subwordsequences,
                            unique_sources_subwordsequences_mask):
                        subword_embedding_weights.append(
                            subword_net(subwordsequences_ctx,
                                        mask_ctx).as_in_context(context[0]))
                    subword_embedding_weights = mx.nd.concat(
                        *subword_embedding_weights, dim=0)

                    # Look up subword embeddings of batch
                    subword_embeddings = mx.nd.Embedding(
                        data=source_subword, weight=subword_embedding_weights,
                        input_dim=subword_embedding_weights.shape[0],
                        output_dim=args.emsize)

                # Look up token embeddings of batch
                if embedding_in is not None:
                    word_embeddings = embedding_in(source)
                    emb_in = subword_embeddings + word_embeddings
                else:
                    assert subword_net is not None
                    emb_in = subword_embeddings

                emb_out = embedding_out(target)

                pred = mx.nd.batch_dot(emb_in, emb_out.swapaxes(1, 2))
                loss = loss_function(pred, label)

            loss.backward()

            # Training of dense params
            if subword_net is not None:
                dense_trainer.step(batch_size=args.batch_size)

            # Training of token level embeddings with sparsity objective
            if embedding_in is not None:
                if i % args.eval_interval == 0:
                    lazy_update = False  # Force eager update before evaluation
                else:
                    lazy_update = True
                emb_in_grad_normalization = mx.nd.sparse.row_sparse_array(
                    (unique_sources_counts.reshape(
                        (-1, 1)), unique_sources_indices), ctx=context[0],
                    dtype=np.float32)
                utils.train_embedding(
                    args, embedding_in.weight.data(context[0]),
                    embedding_in.weight.grad(
                        context[0]), emb_in_grad_normalization,
                    with_sparsity=True, last_update_buffer=last_update_buffer,
                    current_update=current_update, lazy_update=lazy_update)

            # Training of emb_out
            emb_out_grad_normalization = mx.nd.sparse.row_sparse_array(
                (unique_targets_counts.reshape(
                    (-1, 1)), unique_targets_indices), ctx=context[0],
                dtype=np.float32)
            utils.train_embedding(args, embedding_out.weight.data(context[0]),
                                  embedding_out.weight.grad(context[0]),
                                  emb_out_grad_normalization)

            current_update += 1

            if i % args.eval_interval == 0:
                eval_dict = evaluation.evaluate(
                    args, embedding_in, subword_net, vocab, subword_vocab)
                t.set_postfix(**eval_dict)

                # Mxboard
                # Histogram
                if embedding_in is not None:
                    embedding_in_norm = embedding_in.weight.data(
                        ctx=context[0]).as_in_context(
                            mx.cpu()).tostype("default").norm(axis=1)
                    sw.add_histogram(tag='embedding_in_norm',
                                     values=embedding_in_norm,
                                     global_step=current_update, bins=200)
                    embedding_in_grad = embedding_in.weight.grad(
                        ctx=context[0]).as_in_context(
                            mx.cpu()).tostype("default").norm(axis=1)
                    sw.add_histogram(tag='embedding_in_grad',
                                     values=embedding_in_grad,
                                     global_step=current_update, bins=200)
                if subword_net is not None:
                    for k, v in subword_net.collect_params().items():
                        sw.add_histogram(tag=k, values=v.data(ctx=context[0]),
                                         global_step=current_update, bins=200)
                        sw.add_histogram(tag='grad-' + str(k),
                                         values=v.grad(ctx=context[0]),
                                         global_step=current_update, bins=200)
                # Embedding out
                embedding_out_norm = embedding_in.weight.data(
                    ctx=context[0]).as_in_context(
                        mx.cpu()).tostype("default").norm(axis=1)
                sw.add_histogram(tag='embedding_out_norm',
                                 values=embedding_out_norm,
                                 global_step=current_update, bins=200)
                embedding_out_grad = embedding_in.weight.grad(
                    ctx=context[0]).as_in_context(
                        mx.cpu()).tostype("default").norm(axis=1)
                sw.add_histogram(tag='embedding_out_grad',
                                 values=embedding_out_grad,
                                 global_step=current_update, bins=200)

                # Scalars
                sw.add_scalar(tag='loss', value=loss.mean().asscalar(),
                              global_step=current_update)
                for k, v in eval_dict.items():
                    sw.add_scalar(tag=k, value=float(v),
                                  global_step=current_update)

        # Shut down ThreadPoolExecutor
        executor.shutdown()

    sw.close()


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    if not hasattr(mx.nd.sparse, 'dense_division'):
        logging.warning('Mxnet version is not compiled with '
                        'sparse l2_normalization support. '
                        ' Using slow Python implementation.')

    args_ = get_args()
    validate_args(args_)
    setup_logging(args_)

    if not args_.subword_network == 'fasttext':
        train(args_)
    else:
        fasttext.train(args_)
