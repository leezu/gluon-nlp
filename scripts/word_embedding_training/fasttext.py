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

import math
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

import mxnet as mx
import numpy as np
import tqdm
from mxnet import gluon

import data
import evaluation
import utils

try:
    import tqdm
except ImportError:
    tqdm = None


###############################################################################
# Build the model
###############################################################################
def get_model(args, train_dataset, subword_vocab):
    num_tokens = train_dataset.num_tokens
    num_tokens_and_subwords = num_tokens + len(subword_vocab)

    embedding_in = gluon.nn.SparseEmbedding(num_tokens_and_subwords,
                                            args.emsize)
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

    loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()

    embedding_in.hybridize()
    embedding_out.hybridize()

    return embedding_in, embedding_out, loss


###############################################################################
# Training code
###############################################################################
def train(args):
    train_dataset, vocab, subword_vocab = data.get_train_data(args)
    embedding_in, embedding_out, loss_function = get_model(
        args, train_dataset, subword_vocab)
    context = utils.get_context(args)

    sparse_params = list(embedding_in.collect_params().values()) + list(
        embedding_out.collect_params().values())

    # Auxilary states for group lasso objective
    last_update_buffer = mx.nd.zeros(
        (train_dataset.num_tokens + len(subword_vocab), ), ctx=context[0])
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
        for i, (source, target, label, subword_mask) in zip(
                t, executor.map(train_dataset.__getitem__, batches)):
            source = mx.nd.array(source, ctx=context[0])
            target = mx.nd.array(target, ctx=context[0])
            label = mx.nd.array(label, ctx=context[0])
            subword_mask = mx.nd.array(subword_mask, ctx=context[0])

            with mx.autograd.record():
                # Look up subword embeddings and sum reduce
                subword_embeddings = embedding_in(source)
                subword_embeddings_masked = subword_embeddings * \
                    subword_mask.expand_dims(axis=-1)
                emb_in = mx.nd.sum(subword_embeddings_masked, axis=-2)
                emb_out = embedding_out(target)
                pred = mx.nd.batch_dot(emb_in, emb_out.swapaxes(1, 2))
                loss = loss_function(pred, label)

            loss.backward()

            utils.train_embedding(args, embedding_in.weight.data(context[0]),
                                  embedding_in.weight.grad(
                                      context[0]), with_sparsity=True,
                                  last_update_buffer=last_update_buffer,
                                  current_update=current_update)
            current_update += 1
            utils.train_embedding(args, embedding_out.weight.data(context[0]),
                                  embedding_out.weight.grad(context[0]))

            if i % args.eval_interval == 0:
                eval_dict = evaluation.evaluate(args, embedding_in, None,
                                                vocab, subword_vocab)

                t.set_postfix(
                    # TODO print number of grad norm > 0
                    loss=loss.sum().asscalar(),
                    grad=embedding_in.weight.grad(context[0]).as_in_context(
                        mx.cpu()).norm().asscalar(),
                    data=embedding_in.weight.data(
                        ctx=context[0]).as_in_context(
                            mx.cpu()).tostype("default").norm(
                                axis=1).mean().asscalar(),
                    **eval_dict)

        # Force eager gradient update at end of every epoch
        param_data = embedding_in.weight.data(ctx=context[0])
        mx.nd.sparse.sgd_update(
            weight=param_data, grad=mx.nd.sparse.row_sparse_array(
                param_data.shape,
                ctx=context[0]), last_update_buffer=last_update_buffer,
            lr=args.lr, sparsity=args.sparsity_lambda,
            current_update=current_update, out=param_data)

        # Shut down ThreadPoolExecutor
        executor.shutdown()
