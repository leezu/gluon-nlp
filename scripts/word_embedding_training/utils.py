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
"""Utility functions"""

import logging
import time
from contextlib import contextmanager

import mxnet as mx

import sparse_ops


@contextmanager
def print_time(task):
    start_time = time.time()
    logging.info('Starting to {}'.format(task))
    yield
    logging.info('Finished to {} in {} seconds'.format(
        task,
        time.time() - start_time))


def get_context(args):
    if args.gpu is None or args.gpu == '':
        context = [mx.cpu()]
    else:
        context = [mx.gpu(int(i)) for i in args.gpu]
    return context


def train_embedding(args, param_data, param_grad, with_sparsity=False,
                    last_update_buffer=None, current_update=None,
                    lazy_update=True):
    if args.dont_normalize_gradient or (param_grad is None):
        pass
    elif (hasattr(mx.nd.sparse, 'l2_normalization')
          and not args.force_py_op_normalize_gradient):
        norm = mx.nd.sparse.sqrt(
            mx.nd._internal._square_sum(param_grad, axis=1, keepdims=True))
        mx.nd.sparse.l2_normalization(param_grad, norm, out=param_grad)
    else:
        param_grad = mx.nd.Custom(param_grad, op_type='sparse_l2normalization')

    if with_sparsity:  # embedding_in
        assert current_update is not None
        assert param_data.shape[0] == last_update_buffer.shape[0]
        assert last_update_buffer.max().asscalar() < current_update

        if param_grad is None:  # Helper to force eager update with grad
            param_grad = mx.nd.sparse.row_sparse_array(param_data.shape,
                                                       ctx=param_data.context)

        mx.nd.sparse.sgd_update(weight=param_data, grad=param_grad,
                                last_update_buffer=last_update_buffer,
                                lr=args.lr, sparsity=args.sparsity_lambda,
                                current_update=current_update, out=param_data,
                                lazy_update=lazy_update)
    else:
        # TODO make last_update_buffer and current_update optional on mxnet side.
        mx.nd.sparse.sgd_update(
            weight=param_data, grad=param_grad, last_update_buffer=mx.nd.zeros(
                param_data.shape,
                ctx=param_data.context), lr=args.lr, sparsity=0,
            current_update=0, out=param_data, lazy_update=lazy_update)
