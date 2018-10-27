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
from mxnet import gluon


def add_parameters(parser):
    group = parser.add_argument_group('General optimization arguments')
    group.add_argument('--clip-group-gradient-norm', type=float, default=1.0,
                       help='Rescale gradients of groups '
                       'so that their norm does not surpass this.')

    group = parser.add_argument_group('Word level optimization arguments')
    group.add_argument('--optimizer', type=str, default='groupadagrad')
    group.add_argument('--optimizer-eps', type=float)
    group.add_argument('--adagrad-groupwise-lr', action='store_true')
    group.add_argument('--adagrad-decay-states', action='store_true')
    group.add_argument('--adagrad-lazy-decay', action='store_true')
    group.add_argument('--adagrad-decay-factor', type=float, default=0.9)
    group.add_argument('--lr', type=float, default=0.3,
                       help='Learning rate for embeddings matrix.')
    group.add_argument('--lr-schedule', type=str, default='linear',
                       help='Learning rate schedule.')
    group.add_argument('--lr-schedule-step-size', type=int, default=2,
                       help='Step size for step learning rate schedule.')
    group.add_argument('--lr-schedule-step-drop', type=float, default=0.5,
                       help='Drop for step learning rate schedule.')
    group.add_argument(
        '--l2', type=float, default=0,
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
                       default='groupadagrad',
                       help='Optimizer used to train subword network.')
    group.add_argument('--subword-sparse-optimizer-eps', type=float)
    group.add_argument('--subword-sparse-lr', type=float, default=0.3,
                       help='Learning rate for subword embedding network.')
    group.add_argument('--subword-sparse-lr-schedule', type=str,
                       default='linear', help='Learning rate schedule.')
    group.add_argument('--subword-sparse-l2', type=float, default=0,
                       help='Group sparsity regularization scale. '
                       'Parameter is used as multiplier of the '
                       'inverse subword vocabulary size.')


def get_embedding_in_trainer(args, params, num_words):
    # if 'proximal' in args.optimizer.lower():
    #     if not args.ngram_buckets and args.l2 != 0:
    #         warnings.warn('Enabling sparsity regularization {} '
    #                       'without having a subword net. '.format(args.l2))
    #     l2 = args.l2 * 1 / num_words
    #     logging.info('Setting l2 sparsity factor for words '
    #                  'to {}'.format(l2))
    #     kwargs = dict(learning_rate=args.lr, l2_regularization_strength=l2)
    #     if args.optimizer.lower() == 'proximalsgd':
    #         kwargs = dict(
    #             clip_group_gradient_norm=args.clip_group_gradient_norm,
    #             **kwargs)
    #     optimizer = mx.optimizer.Optimizer.create_optimizer(
    #         args.optimizer, **kwargs)
    if args.optimizer.lower() in [
            'sgd', 'adam', 'adagrad', 'ftml', 'groupadagrad'
    ]:
        if args.optimizer_eps:
            optimizer = mx.optimizer.Optimizer.create_optimizer(
                args.optimizer, learning_rate=args.lr, eps=args.optimizer_eps)
        else:
            optimizer = mx.optimizer.Optimizer.create_optimizer(
                args.optimizer, learning_rate=args.lr)
    elif args.optimizer.lower() == 'rmsprop':
        optimizer = mx.optimizer.Optimizer.create_optimizer(
            args.optimizer, learning_rate=args.lr, gamma1=args.adagrad_decay_factor)
    elif args.optimizer.lower() == 'adadelta':
        optimizer = mx.optimizer.Optimizer.create_optimizer(
            args.optimizer)
    else:
        logging.error('Unsupported optimizer')
        sys.exit(1)
    return gluon.Trainer(params, optimizer)


def get_embedding_out_trainer(args, params):
    if args.optimizer.lower() in [
            'sgd', 'adam', 'adagrad', 'ftml', 'groupadagrad'
    ]:
        if args.optimizer_eps:
            optimizer = mx.optimizer.Optimizer.create_optimizer(
                args.optimizer, learning_rate=args.lr, eps=args.optimizer_eps)
        else:
            optimizer = mx.optimizer.Optimizer.create_optimizer(
                args.optimizer, learning_rate=args.lr)
    elif args.optimizer.lower() == 'rmsprop':
        optimizer = mx.optimizer.Optimizer.create_optimizer(
            args.optimizer, learning_rate=args.lr,
            gamma1=args.adagrad_decay_factor)
    elif args.optimizer.lower() == 'adadelta':
        optimizer = mx.optimizer.Optimizer.create_optimizer(args.optimizer)
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
    # if 'proximal' in args.subword_sparse_optimizer.lower():
    #     l2 = args.subword_sparse_l2 * 1 / num_subword_units
    #     logging.info('Setting l2 sparsity factor for subwords '
    #                  'to {}'.format(l2))
    #     kwargs = dict(learning_rate=args.subword_sparse_lr,
    #                   l2_regularization_strength=l2)
    #     if args.subword_sparse_optimizer.lower() == 'proximalsgd':
    #         kwargs = dict(
    #             clip_group_gradient_norm=args.clip_group_gradient_norm,
    #             **kwargs)
    #     optimizer = mx.optimizer.Optimizer.create_optimizer(
    #         args.subword_sparse_optimizer, **kwargs)
    if args.subword_sparse_optimizer.lower() in [
            'sgd', 'adagrad', 'adam', 'ftml', 'groupadagrad'
    ]:
        if args.optimizer_eps:
            optimizer = mx.optimizer.Optimizer.create_optimizer(
                args.optimizer,
                learning_rate=args.subword_sparse_lr,
                eps=args.optimizer_eps)
        else:
            optimizer = mx.optimizer.Optimizer.create_optimizer(
                args.optimizer, learning_rate=args.subword_sparse_lr)
    elif args.optimizer.lower() == 'adadelta':
        optimizer = mx.optimizer.Optimizer.create_optimizer(
            args.optimizer)
    elif args.optimizer.lower() == 'rmsprop':
        optimizer = mx.optimizer.Optimizer.create_optimizer(
            args.optimizer, learning_rate=args.lr, gamma1=args.adagrad_decay_factor)
    else:
        logging.error('Unsupported optimizer')
        sys.exit(1)
    return gluon.Trainer(params, optimizer)


def _get_dense_subword_trainer(args, params):
    if args.subword_dense_optimizer.lower() == 'sgd':
        optimizer = mx.optimizer.Optimizer.create_optimizer(
            args.subword_dense_optimizer, learning_rate=args.subword_dense_lr,
            wd=args.subword_dense_wd, momentum=args.subword_dense_momentum)
    elif args.subword_dense_optimizer.lower() in ['adam', 'adagrad', 'ftml']:
        optimizer = mx.optimizer.Optimizer.create_optimizer(
            args.subword_dense_optimizer, learning_rate=args.subword_dense_lr,
            wd=args.subword_dense_wd)
    else:
        logging.error('Unsupported optimizer')
        sys.exit(1)
    return gluon.Trainer(params, optimizer)
