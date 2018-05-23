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
import warnings

import mxnet as mx


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


def warnings_and_asserst(args):
    if args.word_l2 > 0 and not args.subword_network:
        warnings.warn('Sparsity regularization enabled '
                      'without subword network.')
