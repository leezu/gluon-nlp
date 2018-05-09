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
from concurrent.futures import ThreadPoolExecutor

import mxnet as mx
import numpy as np
from mxnet import gluon

import gluonnlp as nlp
import subword

import data
import evaluation
import sparse_ops
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
    group.add_argument(
        '--subword-network', type=str, default='SubwordCNN',
        help=('Network architecture to infer subword level embeddings. ' +
              str(subword.list_subwordnetworks())))
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
    group.add_argument('--dont-normalize-gradient', action='store_true',
                       help='L2 normalize word embedding gradients per word.')
    group.add_argument(
        '--force-py-op-normalize-gradient', action='store_true',
        help='Always use Python sparse L2 normalization operator.')

    # Logging options
    group = parser.add_argument_group('Logging arguments')
    group.add_argument(
        '--log', type=str, default='results.csv', help='Path to logfile.'
        'Results of evaluation runs are written to there in a CSV format.')

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


class SubwordEmbeddings(gluon.HybridBlock):
    def __init__(self, embedding_dim, length, subword_network, **kwargs):
        super().__init__(**kwargs)

        self.embedding_dim = embedding_dim
        self.length = length

        with self.name_scope():
            if 'rnn' in subword_network.lower():
                self.subword = subword.create(
                    name=subword_network, mode='lstm', length=length,
                    embed_size=32, hidden_size=64, output_size=embedding_dim)
            else:
                self.subword = subword.create(name=subword_network,
                                              embed_size=32,
                                              output_size=embedding_dim)

    def hybrid_forward(self, F, token_bytes):
        # TODO add valid length mask for the subword time pool
        return self.subword(token_bytes)


###############################################################################
# Build the model
###############################################################################
def get_model(args, train_dataset):
    num_tokens = train_dataset.num_tokens

    embedding_in = gluon.nn.SparseEmbedding(num_tokens, args.emsize)
    subword_net = SubwordEmbeddings(embedding_dim=args.emsize,
                                    length=train_dataset.idx_to_bytes.shape[1],
                                    subword_network=args.subword_network)
    embedding_out = gluon.nn.SparseEmbedding(num_tokens, args.emsize)

    context = utils.get_context(args)
    embeddings_context = [context[0]]
    if args.normalized_initialization:
        embedding_in.initialize(
            mx.init.Uniform(scale=1 / args.emsize), ctx=embeddings_context)
        embedding_out.initialize(
            mx.init.Uniform(scale=1 / args.emsize), ctx=embeddings_context)
    else:
        embedding_in.initialize(mx.init.Uniform(), ctx=embeddings_context)
        embedding_out.initialize(mx.init.Uniform(), ctx=embeddings_context)
    subword_net.initialize(mx.init.Xavier(), ctx=context)

    loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()

    embedding_in.hybridize()
    embedding_out.hybridize()
    # subword_net.hybridize()

    return embedding_in, embedding_out, subword_net, loss


###############################################################################
# Training code
###############################################################################
def train(args):
    train_dataset, vocab, subword_vocab = data.get_train_data(args)
    embedding_in, embedding_out, subword_net, loss_function = get_model(
        args, train_dataset)
    context = utils.get_context(args)

    sparse_params = list(embedding_in.collect_params().values()) + list(
        embedding_out.collect_params().values())
    dense_params = list(subword_net.collect_params().values())

    dense_trainer = gluon.Trainer(dense_params, 'sgd', {
        'learning_rate': args.lr,
        'momentum': 0.1
    })

    # Auxilary states for group lasso objective
    last_update_buffer = mx.nd.zeros((train_dataset.num_tokens, ),
                                     ctx=context[0])
    current_update = 1

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
        for i, (source, target, label, token_bytes, source_subword) in zip(
                t, executor.map(train_dataset.__getitem__, batches)):
            mx.nd.waitall()

            # Load data for training embedding matrix to context[0]
            source = gluon.utils.split_and_load(source, [context[0]])[0]
            target = gluon.utils.split_and_load(target, [context[0]])[0]
            label = gluon.utils.split_and_load(label, [context[0]])[0]

            # Load indices for looking up subword embedding to context[0]
            source_subword = gluon.utils.split_and_load(
                source_subword, [context[0]])[0]

            # Split and load subword info to all GPUs for accelerated computation
            token_bytes = gluon.utils.split_and_load(
                token_bytes, context, batch_axis=1, even_split=False)

            with mx.autograd.record():
                # Compute subword embeddings from subword info (byte sequences)
                subword_embedding_weights = []
                for token_bytes_ctx in token_bytes:
                    subword_embedding_weights.append(
                        subword_net(token_bytes_ctx).as_in_context(context[0]))
                subword_embedding_weights = mx.nd.concat(
                    *subword_embedding_weights, dim=0)

                # Look up subword embeddings of batch
                subword_embeddings = mx.nd.Embedding(
                    data=source_subword, weight=subword_embedding_weights,
                    input_dim=subword_embedding_weights.shape[0],
                    output_dim=args.emsize)

                # Look up token embeddings of batch
                word_embeddings = embedding_in(source)

                emb_in = subword_embeddings + word_embeddings
                emb_out = embedding_out(target)

                pred = mx.nd.batch_dot(emb_in, emb_out.swapaxes(1, 2))
                loss = loss_function(pred, label)

            loss.backward()

            # Training of dense params
            dense_trainer.step(batch_size=args.batch_size)

            # Training of sparse params
            for param_i, param in enumerate(sparse_params):
                if param.grad_req == 'null':
                    continue

                # Update of sparse params
                for device_param, device_grad in zip(param.list_data(),
                                                     param.list_grad()):
                    if args.dont_normalize_gradient:
                        pass
                    elif (hasattr(mx.nd.sparse, 'l2_normalization')
                          and not args.force_py_op_normalize_gradient):
                        norm = mx.nd.sparse.sqrt(
                            mx.nd._internal._square_sum(
                                device_grad, axis=1, keepdims=True))
                        mx.nd.sparse.l2_normalization(device_grad, norm,
                                                      out=device_grad)
                    else:
                        device_grad = mx.nd.Custom(
                            device_grad, op_type='sparse_l2normalization')

                    mx.nd.sparse.sgd_update(
                        weight=device_param, grad=device_grad,
                        last_update_buffer=last_update_buffer, lr=args.lr,
                        sparsity=args.sparsity_lambda,
                        current_update=current_update, out=device_param)
                    current_update += 1

            if i % args.eval_interval == 0:
                eval_dict = evaluation.evaluate(
                    args, embedding_in, subword_net, vocab, subword_vocab)

                t.set_postfix(
                    # TODO print number of grad norm > 0
                    loss=loss.sum().asscalar(),
                    grad=embedding_in.weight.grad(context[0]).as_in_context(
                        mx.cpu()).norm().asscalar(),
                    dense_grad=sum(
                        p.grad(ctx=context[0]).norm()
                        for p in dense_params).asscalar(),
                    data=embedding_in.weight.data(
                        ctx=context[0]).as_in_context(
                            mx.cpu()).tostype("default").norm(
                                axis=1).mean().asscalar(),
                    **eval_dict)

        # Force eager gradient update at end of every epoch
        for device_param, device_grad in zip(param.list_data(),
                                             param.list_grad()):
            mx.nd.sparse.sgd_update(
                weight=device_param, grad=mx.nd.sparse.row_sparse_array(
                    device_grad.shape,
                    ctx=context[0]), last_update_buffer=last_update_buffer,
                lr=args.lr, sparsity=args.sparsity_lambda,
                current_update=current_update, out=device_param)

        # Shut down ThreadPoolExecutor
        executor.shutdown()


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    if not hasattr(mx.nd.sparse, 'l2_normalization'):
        logging.warning('Mxnet version is not compiled with '
                        'sparse l2_normalization support. '
                        ' Using slow Python implementation.')

    args_ = get_args()
    validate_args(args_)

    train(args_)
