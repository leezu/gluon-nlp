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
import os
import tempfile
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


def train_embedding(args, param_data, param_grad, grad_normalization=None,
                    with_sparsity=False, last_update_buffer=None,
                    current_update=None, lazy_update=True):
    if (not args.normalize_gradient
            or args.normalize_gradient.lower() == 'none'
            or (param_grad is None)):
        pass
    else:
        if args.normalize_gradient.lower() == 'count':
            assert grad_normalization is not None
            norm = grad_normalization
        elif args.normalize_gradient.lower() == 'l2':
            norm = mx.nd.sparse.sqrt(
                mx.nd._internal._square_sum(param_grad, axis=1, keepdims=True))
        else:
            raise NotImplementedError

        if (hasattr(mx.nd.sparse, 'dense_division')
                and not args.force_py_op_normalize_gradient):
            mx.nd.sparse.dense_division(param_grad, norm, out=param_grad)
        else:
            param_grad = mx.nd.Custom(param_grad, norm,
                                      op_type='dense_division')

    if with_sparsity:  # embedding_in
        assert current_update is not None
        assert param_data.shape[0] == last_update_buffer.shape[0]
        assert last_update_buffer.max().asscalar() < current_update

        if param_grad is None:  # Helper to force eager update with grad
            param_grad = mx.nd.sparse.row_sparse_array(param_data.shape,
                                                       ctx=param_data.context)

        mx.nd.sparse.proximal_sgd_update(
            weight=param_data, grad=param_grad,
            last_update_buffer=last_update_buffer, lr=args.embeddings_lr,
            sparsity=args.sparsity_lambda, current_update=current_update,
            out=param_data, lazy_update=lazy_update)
    else:
        mx.nd.sparse.sgd_update(weight=param_data, grad=param_grad,
                                lr=args.embeddings_lr, out=param_data,
                                lazy_update=lazy_update)


def _get_tempfilename(directory):
    f, path = tempfile.mkstemp(dir=directory)
    os.close(f)
    return path


def save_params(args, embedding_in, embedding_out, subword_net,
                global_step=''):
    # write to temporary file; use os.replace
    if embedding_in is not None:
        p = _get_tempfilename(args.logdir)
        embedding_in.collect_params().save(p)
        os.replace(p, os.path.join(args.logdir, 'embedding_in'))
    if embedding_out is not None:
        p = _get_tempfilename(args.logdir)
        embedding_out.collect_params().save(p)
        os.replace(p, os.path.join(args.logdir, 'embedding_out'))
    if subword_net is not None:
        p = _get_tempfilename(args.logdir)
        subword_net.collect_params().save(p)
        os.replace(p, os.path.join(args.logdir, 'subword_net'))
