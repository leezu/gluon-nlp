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

import logging
import sys
import warnings

import mxnet as mx
import numpy as np
from mxnet import gluon
from mxnet.ndarray import NDArray, full, proximal_sgd_update, sgd_update

import utils


def add_parameters(parser):
    group = parser.add_argument_group('Word level optimization arguments')
    group.add_argument('--word-optimizer', type=str, default='proximalsgd')
    group.add_argument('--word-lr', type=float, default=0.1,
                       help='Learning rate for embeddings matrix.')
    group.add_argument('--word-lr-schedule', type=str, default='linear',
                       help='Learning rate schedule.')
    group.add_argument(
        '--word-l2', type=float, default=1,
        help='Group sparsity regularization scale. '
        'Parameter is used as multiplier of the inverse vocabulary size.')

    group = parser.add_argument_group(
        'Dense subword network optimization arguments')
    group.add_argument('--subword-dense-optimizer', type=str,
                       default='adagrad',
                       help='Optimizer used to train subword network.')
    group.add_argument('--subword-dense-lr', type=float, default=0.01,
                       help='Learning rate for subword embedding network.')
    group.add_argument('--subword-dense-lr-schedule', type=str,
                       default='linear', help='Learning rate schedule.')
    group.add_argument('--subword-dense-wd', type=float, default=1.2e-6,
                       help='Weight decay for subword embedding network.')
    group.add_argument('--subword-dense-momentum', type=float, default=0.9,
                       help='Momentum for subword-dense-optimizer.')

    group = parser.add_argument_group(
        'Sparse subword network optimization arguments')
    group.add_argument('--subword-sparse-optimizer', type=str,
                       default='proximalsgd',
                       help='Optimizer used to train subword network.')
    group.add_argument('--subword-sparse-lr', type=float, default=0.1,
                       help='Learning rate for subword embedding network.')
    group.add_argument('--subword-sparse-lr-schedule', type=str,
                       default='linear', help='Learning rate schedule.')
    group.add_argument('--subword-sparse-l2', type=float, default=1,
                       help='Group sparsity regularization scale. '
                       'Parameter is used as multiplier of the '
                       'inverse subword vocabulary size.')


def get_embedding_in_trainer(args, params, num_words):
    if args.word_optimizer.lower() == 'proximalsgd':
        if not args.subword_network.lower() and args.word_l2 != 0:
            warnings.warn('Enabling sparsity regularization {} '
                          'without having a subword net. '.format(
                              args.word_l2))
        l2 = args.word_l2 * 1 / num_words
        logging.info('Setting l2 sparsity factor for words '
                     'to {}'.format(l2))
        optimizer = mx.optimizer.Optimizer.create_optimizer(
            args.word_optimizer, learning_rate=args.word_lr, l2=l2)
    elif args.word_optimizer.lower() == 'sgd':
        optimizer = mx.optimizer.Optimizer.create_optimizer(
            args.word_optimizer, learning_rate=args.word_lr)
    elif args.word_optimizer.lower() == 'adagrad':
        optimizer = mx.optimizer.Optimizer.create_optimizer(
            args.word_optimizer, learning_rate=args.word_lr)
    else:
        logging.error('Unsupported optimizer')
        sys.exit(1)
    return gluon.Trainer(params, optimizer)


def get_embedding_out_trainer(args, params):
    if args.word_optimizer.lower() in ['proximalsgd', 'sgd']:
        # Ignore group sparsity for context matrix
        optimizer = mx.optimizer.Optimizer.create_optimizer(
            'sgd', learning_rate=args.word_lr)
    elif args.word_optimizer.lower() in ['adagrad']:
        optimizer = mx.optimizer.Optimizer.create_optimizer(
            'adagrad', learning_rate=args.word_lr)
    else:
        logging.error('Unsupported optimizer')
        sys.exit(1)
    return gluon.Trainer(params, optimizer)


def get_subword_trainer(args, params, num_subword_units):
    """Parase args depending on subwort network and return trainer."""
    if args.subword_network.lower() in ['sumreduce', 'meanreduce', 'fasttext']:
        return _get_sparse_subword_trainer(args, params, num_subword_units)
    else:
        return _get_dense_subword_trainer(args, params)


def _get_sparse_subword_trainer(args, params, num_subword_units):
    if args.subword_sparse_optimizer.lower() == 'proximalsgd':
        l2 = args.subword_sparse_l2 * 1 / num_subword_units
        logging.info('Setting l2 sparsity factor for subwords '
                     'to {}'.format(l2))
        optimizer = mx.optimizer.Optimizer.create_optimizer(
            args.subword_sparse_optimizer,
            learning_rate=args.subword_sparse_lr, l2=l2)
    elif args.subword_sparse_optimizer.lower() == 'sgd':
        optimizer = mx.optimizer.Optimizer.create_optimizer(
            args.subword_sparse_optimizer,
            learning_rate=args.subword_sparse_lr)
    elif args.word_optimizer.lower() in ['adagrad']:
        optimizer = mx.optimizer.Optimizer.create_optimizer(
            'adagrad', learning_rate=args.subword_sparse_lr)
    else:
        logging.error('Unsupported optimizer')
        sys.exit(1)
    return gluon.Trainer(params, optimizer)


def _get_dense_subword_trainer(args, params):
    if args.subword_dense_optimizer.lower() == 'sgd':
        optimizer = mx.optimizer.Optimizer.create_optimizer(
            args.subword_dense_optimizer, learning_rate=args.subword_dense_lr,
            wd=args.subword_dense_wd, momentum=args.subword_dense_momentum)
    elif args.subword_dense_optimizer.lower() in ['adam', 'adagrad']:
        optimizer = mx.optimizer.Optimizer.create_optimizer(
            args.subword_dense_optimizer, learning_rate=args.subword_dense_lr,
            wd=args.subword_dense_wd)
    else:
        logging.error('Unsupported optimizer')
        sys.exit(1)
    return gluon.Trainer(params, optimizer)


def normalize_sparse_grads(args, embedding_block, unique_counts=None,
                           unique_indices=None):
    context = utils.get_context(args)
    assert len(context) == 1

    param_grad = embedding_block.weight.grad(context[0])

    if args.normalize_gradient.lower() == 'none':
        norm = None
    elif args.normalize_gradient.lower() == 'count':
        norm = mx.nd.sparse.row_sparse_array((unique_counts.reshape(
            (-1, 1)), unique_indices), ctx=context[0], dtype=np.float32)
    elif args.normalize_gradient.lower() == 'l2':
        norm = mx.nd.sparse.sqrt(
            mx.nd._internal._square_sum(param_grad, axis=1, keepdims=True))
    else:
        raise NotImplementedError

    if norm is not None:
        if (hasattr(mx.nd.sparse, 'dense_division')
                and not args.force_py_op_normalize_gradient):
            mx.nd.sparse.dense_division(param_grad, norm, out=param_grad)
        else:
            param_grad = mx.nd.Custom(param_grad, norm,
                                      op_type='dense_division')


# pylint: disable=line-too-long
@mx.optimizer.register
class ProximalSGD(mx.optimizer.Optimizer):
    """A Proximal SGD optimizer.

    Standard updates are applied by::

        rescaled_grad = lr * rescale_grad * clip(grad, clip_gradient) + wd * weight
        state = momentum * state + rescaled_grad
        weight = weight - state

    Then a proximal operator is executed.

    For details of the update algorithm see
    :class:`~mxnet.ndarray.proximal_sgd_update`.

    This optimizer accepts the following parameters in addition to those accepted
    by :class:`.Optimizer`.

    Parameters
    ----------
    l2 : float
       Sparsity lambda for l2.
    lazy_update : bool, optional
       Default is True. If True, lazy updates are applied \
       if the storage types of weight and grad are both ``row_sparse``.
    """

    def __init__(self, l2=0.0, lazy_update=True, **kwargs):
        super(ProximalSGD, self).__init__(**kwargs)
        self.l2 = l2
        self.lazy_update = lazy_update

    def create_state(self, index, weight):
        last_update_buffer = None
        if self.l2 != 0.0:
            last_update_buffer = full((weight.shape[0], ), self.num_update,
                                      weight.context)
        return last_update_buffer

    def _update_impl(self, index, weight, grad, state):
        assert (isinstance(weight, NDArray))
        assert (isinstance(grad, NDArray))
        self._update_count(index)
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        assert wd == 0
        assert self.clip_gradient is None

        kwargs = {'rescale_grad': self.rescale_grad}

        if state is not None:
            proximal_sgd_update(
                weight, grad, out=weight, lazy_update=self.lazy_update, lr=lr,
                last_update_buffer=state, current_update=self.num_update,
                sparsity=self.l2)
        else:
            sgd_update(weight, grad, out=weight, lazy_update=self.lazy_update,
                       lr=lr, **kwargs)

    def update(self, index, weight, grad, state):
        self._update_impl(index, weight, grad, state)
