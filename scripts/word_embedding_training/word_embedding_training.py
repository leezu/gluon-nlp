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
import sys
import tempfile
import os

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
        '--subword-network', type=str, default='', nargs='?',
        help=('Network architecture to infer subword level embeddings. ' +
              str(subword.list_subwordnetworks()) +
              ' , fasttext or empty to disable'))
    group.add_argument('--subword-function', type=str, default='character')
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
    group.add_argument('--sparsity-lambda', type=float, default=0.001,
                       help='Initial learning rate')
    group.add_argument('--epochs', type=int, default=5, help='Epoch limit')
    group.add_argument('--gpu', type=int, nargs='+',
                       help=('Number (index) of GPU to run on, e.g. 0. '
                             'If not specified, uses CPU.'))
    group.add_argument('--dont-hybridize', action='store_true',
                       help='Disable hybridization of gluon HybridBlocks.')
    group.add_argument('--no-normalize-embeddings', action='store_true',
                       help='Normalize the word embeddings row-wise.')
    group.add_argument('--normalize-gradient', type=str, default='count',
                       help='Normalize the word embedding gradient row-wise. '
                       'Supported are [None, count, L2].')
    group.add_argument(
        '--force-py-op-normalize-gradient', action='store_true',
        help='Always use Python sparse L2 normalization operator.')
    group.add_argument('--use-threaded-data-workers', action='store_true',
                       help='Enable threaded data pre-fetching.')
    group.add_argument('--num-data-workers', type=int, default=5,
                       help='Number of threads to preload data.')

    # Optimization options
    group = parser.add_argument_group('Optimization arguments')
    group.add_argument('--embeddings-lr', type=float, default=0.1,
                       help='Learning rate for embeddings matrix.')
    group.add_argument('--dense-lr', type=float, default=0.1,
                       help='Learning rate for subword embedding network.')
    group.add_argument('--dense-wd', type=float, default=1.2e-6,
                       help='Weight decay for subword embedding network.')
    group.add_argument('--dense-optimizer', type=str, default='adam',
                       help='Optimizer used to train subword network.')
    group.add_argument('--dense-momentum', type=float, default=0.9,
                       help='Momentum for dense-optimizer, if supported.')
    # Logging options
    group = parser.add_argument_group('Logging arguments')
    group.add_argument('--logdir', type=str, default=None,
                       help='Directory to store logs in.'
                       'Tensorboard compatible logs are stored there. '
                       'Defaults to a random directory in ./logs')

    # Add further argument groups
    subword.add_subword_parameters_to_parser(parser)
    data.add_parameters(parser)

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
    elif not os.path.isdir(args.logdir):
        os.makedirs(args.logdir)
    elif os.path.isfile(args.logdir):
        raise RuntimeError('{} is a file.'.format(args.logdir))

    logging.info('Logging to {}'.format(args.logdir))


###############################################################################
# Build the model
###############################################################################
def get_model(args, train_dataset, subword_vocab):
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
        subword_net = subword.create(name=args.subword_network, args=args,
                                     vocab_size=len(subword_vocab))
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
        args, train_dataset, subword_vocab)
    context = utils.get_context(args)

    if subword_net is not None:
        if args.dense_optimizer.lower() == 'sgd':
            dense_trainer = gluon.Trainer(
                subword_net.collect_params(), args.dense_optimizer, {
                    'learning_rate': args.dense_lr,
                    'wd': args.dense_wd,
                    'momentum': args.dense_momentum
                })
        elif args.dense_optimizer.lower() in ['adam', 'adagrad']:
            dense_trainer = gluon.Trainer(subword_net.collect_params(),
                                          args.dense_optimizer, {
                                              'learning_rate': args.dense_lr,
                                              'wd': args.dense_wd,
                                          })
        else:
            logging.error('Unsupported optimizer')
            sys.exit(1)

    # Auxilary states for group lasso objective
    last_update_buffer = mx.nd.zeros((train_dataset.num_tokens, ),
                                     ctx=context[0])
    current_update = 1

    # Logging writer
    sw = SummaryWriter(logdir=args.logdir)

    indices = np.arange(len(train_dataset))
    for epoch in range(args.epochs):
        np.random.shuffle(indices)
        with utils.print_time('create batch indices'):
            batches = [
                indices[i:i + args.batch_size]
                for i in range(0, len(indices), args.batch_size)
            ]

        if tqdm is not None:
            t = tqdm.trange(len(batches), smoothing=1)
        else:
            t = range(len(batches))

        for i, batch_idx in zip(t, batches):
            batch = train_dataset[batch_idx]
            (source, target, label, unique_sources_indices,
             unique_sources_counts, unique_sources_subwordsequences,
             source_subword, unique_sources_subwordsequences_mask,
             unique_targets_indices, unique_targets_counts) = batch

            unique_sources_subwordsequences_last_valid = \
                (unique_sources_subwordsequences_mask == 0).argmax(axis=1) - 1
            unique_sources_subwordsequences_last_valid[
                unique_sources_subwordsequences_last_valid ==
                -1] = unique_sources_subwordsequences_mask.shape[1] - 1

            # Load data for training embedding matrix to context[0]
            source = mx.nd.array(source, ctx=context[0])
            target = mx.nd.array(target, ctx=context[0])
            label = mx.nd.array(label, ctx=context[0])

            # Load indices for looking up subword embedding to context[0]
            source_subword = mx.nd.array(source_subword, ctx=context[0])

            # Split and load subword info to all GPUs for accelerated computation
            assert unique_sources_subwordsequences.shape == \
                unique_sources_subwordsequences_mask.shape
            unique_token_subwordsequences = gluon.utils.split_and_load(
                unique_sources_subwordsequences, context, even_split=False)
            unique_sources_subwordsequences_mask = gluon.utils.split_and_load(
                unique_sources_subwordsequences_mask, context,
                even_split=False)
            unique_sources_subwordsequences_last_valid = \
                gluon.utils.split_and_load(
                    unique_sources_subwordsequences_last_valid,
                    context, even_split=False)

            with mx.autograd.record():
                if subword_net is not None:
                    # Compute subword embeddings from subword info (byte sequences)
                    subword_embedding_weights = []
                    attention_regularizers = []
                    for subwordsequences_ctx, mask_ctx, last_valid_ctx in zip(
                            unique_token_subwordsequences,
                            unique_sources_subwordsequences_mask,
                            unique_sources_subwordsequences_last_valid):
                        out, att_weights = subword_net(
                            subwordsequences_ctx, mask_ctx, last_valid_ctx)
                        if att_weights is not None:
                            attention_regularizer = mx.nd.sqrt(
                                mx.nd.sum((mx.nd.batch_dot(
                                    att_weights, att_weights.swapaxes(1, 2)
                                ) - mx.nd.eye(
                                    args.
                                    subwordrnn_self_attention_num_attention,
                                    ctx=att_weights.context))**2))
                            attention_regularizers.append(
                                attention_regularizer.as_in_context(
                                    context[0]))

                        # TODO check if the gradient is actually passed when switching device
                        subword_embedding_weights.append(
                            out.as_in_context(context[0]))
                    subword_embedding_weights = mx.nd.concat(
                        *subword_embedding_weights, dim=0)

                    # TODO remove
                    if attention_regularizers:
                        attention_regularizers = mx.nd.sum(
                            mx.nd.concat(*attention_regularizers, dim=0))
                    else:
                        attention_regularizers = None

                    # Look up subword embeddings of batch
                    subword_embeddings = mx.nd.Embedding(
                        data=source_subword, weight=subword_embedding_weights,
                        input_dim=subword_embedding_weights.shape[0],
                        output_dim=args.emsize)

                # Look up token embeddings of batch
                if embedding_in is not None:
                    word_embeddings = embedding_in(source)
                    if subword_net is not None:
                        # Warning: L2Normalization is only correct as emb_in
                        # only has 1 element in middle dim (bs, 1, emsize)
                        if not args.no_normalize_embeddings:
                            emb_in = mx.nd.L2Normalization(subword_embeddings)\
                                + mx.nd.L2Normalization(word_embeddings)
                        else:
                            emb_in = subword_embeddings + word_embeddings
                    else:
                        if not args.no_normalize_embeddings:
                            emb_in = mx.nd.L2Normalization(word_embeddings)
                        else:
                            emb_in = word_embeddings
                else:
                    assert subword_net is not None
                    if not args.no_normalize_embeddings:
                        emb_in = mx.nd.L2Normalization(subword_embeddings)
                    else:
                        emb_in = subword_embeddings

                emb_out = embedding_out(target)
                if not args.no_normalize_embeddings:
                    emb_out_shape = emb_out.shape
                    emb_out = mx.nd.L2Normalization(
                        emb_out.reshape((-1,
                                         args.emsize))).reshape(emb_out_shape)

                pred = mx.nd.batch_dot(emb_in, emb_out.swapaxes(1, 2))
                if (subword_net is not None
                        and attention_regularizers is not None):
                    loss = loss_function(
                        pred, label) + (args.attention_regularizer_lambda *
                                        mx.nd.sum(attention_regularizers))
                else:
                    loss = loss_function(pred, label)

            loss.backward()

            # Training of dense params
            if subword_net is not None:
                # dense_trainer.step(batch_size=args.batch_size)  # TODO
                dense_trainer.step(batch_size=1)

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
                with utils.print_time('mx.nd.waitall()'):
                    mx.nd.waitall()

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

                        # Predicted word embeddings
                        sw.add_histogram(
                            tag='subword_embedding_in_norm',
                            values=subword_embedding_weights.norm(axis=1),
                            global_step=current_update, bins=200)

                # Embedding out
                embedding_out_norm = embedding_out.weight.data(
                    ctx=context[0]).as_in_context(
                        mx.cpu()).tostype("default").norm(axis=1)
                sw.add_histogram(tag='embedding_out_norm',
                                 values=embedding_out_norm,
                                 global_step=current_update, bins=200)
                embedding_out_grad = embedding_out.weight.grad(
                    ctx=context[0]).as_in_context(
                        mx.cpu()).tostype("default").norm(axis=1)
                sw.add_histogram(tag='embedding_out_grad',
                                 values=embedding_out_grad,
                                 global_step=current_update, bins=200)

                # Scalars
                sw.add_scalar(tag='loss', value=loss.mean().asscalar(),
                              global_step=current_update)

                eval_dict = evaluation.evaluate(
                    args, embedding_in, subword_net, vocab, subword_vocab, sw)
                for k, v in eval_dict.items():
                    sw.add_scalar(tag=k, value=float(v),
                                  global_step=current_update)

                sw.flush()

                # Save params after evaluation
                utils.save_params(args, embedding_in, embedding_out,
                                  subword_net, current_update)

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
