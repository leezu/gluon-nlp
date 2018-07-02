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
"""Trainers
===========

Custom trainers.

"""

import logging
import sys
import warnings

import mxnet as mx
import numpy as np
from mxnet import gluon
from mxnet.ndarray import (NDArray, zeros, full, clip, sqrt, square, sparse,
                           proximal_sgd_update, sgd_update,
                           proximal_adagrad_update)

import utils


def add_parameters(parser):
    group = parser.add_argument_group('General optimization arguments')
    group.add_argument('--clip-group-gradient-norm', type=float, default=1.0,
                       help='Rescale gradients of groups '
                       'so that their norm does not surpass this.')

    group = parser.add_argument_group('Word level optimization arguments')
    group.add_argument('--optimizer', type=str, default='proximalsgd')
    group.add_argument('--lr', type=float, default=0.1,
                       help='Learning rate for embeddings matrix.')
    group.add_argument('--lr-schedule', type=str, default='linear',
                       help='Learning rate schedule.')
    group.add_argument(
        '--l2', type=float, default=1,
        help='Group sparsity regularization scale. '
        'Parameter is used as multiplier of the inverse vocabulary size.')

    group = parser.add_argument_group(
        'Dense subword network optimization arguments')
    group.add_argument('--subword-dense-optimizer', type=str,
                       default='adam',
                       help='Optimizer used to train subword network.')
    group.add_argument('--subword-dense-lr', type=float, default=1e-5,
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
    if 'proximal' in args.optimizer.lower():
        if not args.ngram_buckets and args.l2 != 0:
            warnings.warn('Enabling sparsity regularization {} '
                          'without having a subword net. '.format(args.l2))
        l2 = args.l2 * 1 / num_words
        logging.info('Setting l2 sparsity factor for words '
                     'to {}'.format(l2))
        kwargs = dict(learning_rate=args.lr, l2_regularization_strength=l2)
        if args.optimizer.lower() == 'proximalsgd':
            kwargs = dict(
                clip_group_gradient_norm=args.clip_group_gradient_norm,
                **kwargs)
        optimizer = mx.optimizer.Optimizer.create_optimizer(
            args.optimizer, **kwargs)
    elif args.optimizer.lower() == 'sgd':
        optimizer = mx.optimizer.Optimizer.create_optimizer(
            args.optimizer, learning_rate=args.lr)
    elif args.optimizer.lower() == 'adagrad':
        optimizer = mx.optimizer.Optimizer.create_optimizer(
            args.optimizer, learning_rate=args.lr)
    else:
        logging.error('Unsupported optimizer')
        sys.exit(1)
    return gluon.Trainer(params, optimizer)


def get_embedding_out_trainer(args, params):
    # Ignore group sparsity for context matrix
    if 'sgd' in args.optimizer.lower():
        optimizer = mx.optimizer.Optimizer.create_optimizer(
            'sgd', learning_rate=args.lr)
    elif 'adagrad' in args.optimizer.lower():
        optimizer = mx.optimizer.Optimizer.create_optimizer(
            'adagrad', learning_rate=args.lr)
    else:
        logging.error('Unsupported optimizer')
        sys.exit(1)
    return gluon.Trainer(params, optimizer)


def get_subword_trainer(args, params, num_subword_units):
    """Parase args depending on subwort network and return trainer."""
    if args.subword_network.lower() == 'fasttext':
        return _get_sparse_subword_trainer(args, params, num_subword_units)
    else:
        return _get_dense_subword_trainer(args, params)


def _get_sparse_subword_trainer(args, params, num_subword_units):
    if 'proximal' in args.subword_sparse_optimizer.lower():
        l2 = args.subword_sparse_l2 * 1 / num_subword_units
        logging.info('Setting l2 sparsity factor for subwords '
                     'to {}'.format(l2))
        kwargs = dict(learning_rate=args.subword_sparse_lr,
                      l2_regularization_strength=l2)
        if args.subword_sparse_optimizer.lower() == 'proximalsgd':
            kwargs = dict(
                clip_group_gradient_norm=args.clip_group_gradient_norm,
                **kwargs)
        optimizer = mx.optimizer.Optimizer.create_optimizer(
            args.subword_sparse_optimizer, **kwargs)
    elif args.subword_sparse_optimizer.lower() in ['sgd', 'adagrad']:
        optimizer = mx.optimizer.Optimizer.create_optimizer(
            args.subword_sparse_optimizer,
            learning_rate=args.subword_sparse_lr)
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
    clip_group_gradient_norm : float, optional
       Rescale gradients of each group so that their norm does not surpass
       clip_group_gradient_norm
    l2_regularization_strength : float, default 0.0
       Strength of L2 regularization.
    lazy_update : bool, optional
       Default is True. If True, lazy updates are applied \
       if the storage types of weight and grad are both ``row_sparse``.

    """

    def __init__(self, clip_group_gradient_norm=None,
                 l2_regularization_strength=0.0, lazy_update=True, **kwargs):
        super(ProximalSGD, self).__init__(**kwargs)
        self.clip_group_gradient_norm = clip_group_gradient_norm
        self.l2_regularization_strength = l2_regularization_strength
        self.lazy_update = lazy_update

    def create_state(self, index, weight):
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

        kwargs = dict(
            rescale_grad=self.rescale_grad, lr=lr,
            l2_regularization_strength=self.l2_regularization_strength)
        if self.clip_gradient:
            kwargs['clip_gradient'] = self.clip_gradient
        if self.clip_group_gradient_norm:
            kwargs['clip_group_gradient_norm'] = self.clip_group_gradient_norm

        proximal_sgd_update(
            weight, grad, out=weight, lazy_update=self.lazy_update,
            last_update_buffer=state, current_update=self.num_update, **kwargs)

    def update(self, index, weight, grad, state):
        self._update_impl(index, weight, grad, state)


# pylint: disable=line-too-long
@mx.optimizer.register
class ProximalAdagrad(mx.optimizer.Optimizer):
    """A Proximal Adagrad optimizer.

    Standard updates as for Adagrad are applied.

    Then a proximal operator is executed.

    For details of the update algorithm see
    :class:`~mxnet.ndarray.proximal_adagrad_update`.

    This optimizer accepts the following parameters in addition to those
    accepted by :class:`.Optimizer`.

    Parameters
    ----------
    l2_regularization_strength : float
       Strength of L2 regularization.
    lazy_update : bool, optional
       Default is True. If True, lazy updates are applied \
       if the storage types of weight and grad are both ``row_sparse``.

    """

    def __init__(self, l2_regularization_strength=0.0, lazy_update=True,
                 float_stable_epsilon=1e-5, bisection_epsilon=1e-3, **kwargs):
        super(ProximalAdagrad, self).__init__(**kwargs)
        self.l2_regularization_strength = l2_regularization_strength
        self.lazy_update = lazy_update
        self.float_stable_eps = float_stable_epsilon
        self.bisection_eps = bisection_epsilon

    def create_state(self, index, weight):
        history = zeros(weight.shape, weight.context, stype=weight.stype)
        last_update_buffer = None
        if self.l2_regularization_strength != 0.0:
            last_update_buffer = full(
                shape=(weight.shape[0], ), val=self.num_update,
                ctx=weight.context)
        return [history, last_update_buffer]

    def _update_impl(self, index, weight, grad, state):
        assert (isinstance(weight, NDArray))
        assert (isinstance(grad, NDArray))
        self._update_count(index)
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        assert wd == 0

        is_sparse = grad.stype == 'row_sparse'
        history = state[0]
        last_update_buffer = state[1]
        if last_update_buffer is not None:
            proximal_adagrad_update(
                weight, grad, history, out=weight,
                last_update_buffer=last_update_buffer,
                lazy_update=self.lazy_update, lr=lr,
                current_update=self.num_update,
                l2_regularization_strength=self.l2_regularization_strength,
                rescale_grad=self.rescale_grad,
                float_stable_epsilon=self.float_stable_eps,
                bisection_epsilon=self.bisection_eps)
        elif is_sparse:
            sparse.adagrad_update(weight, grad, history, out=weight, lr=lr,
                                  wd=wd, rescale_grad=self.rescale_grad,
                                  epsilon=self.float_stable_eps)
        else:
            grad = grad * self.rescale_grad
            if self.clip_gradient is not None:
                grad = clip(grad, -self.clip_gradient, self.clip_gradient)
            history[:] += square(grad)
            div = grad / sqrt(history + self.float_stable_eps)
            weight[:] += (div + weight * wd) * -lr

    def update(self, index, weight, grad, state):
        self._update_impl(index, weight, grad, state)
