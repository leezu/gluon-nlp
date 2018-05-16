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

import mxnet as mx
import numpy as np
from mxboard import SummaryWriter
import tqdm
from mxnet import gluon

import data
import evaluation
import utils
import bounded_executor

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

    if not args.dont_hybridize:
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

    # Auxilary states for group lasso objective
    last_update_buffer = mx.nd.zeros(
        (train_dataset.num_tokens + len(subword_vocab), ), ctx=context[0])
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

        if args.use_threaded_data_workers:
            executor = bounded_executor.BoundedExecutor(
                bound=100, max_workers=args.num_data_workers)
            batches = executor.map(train_dataset.__getitem__, batches)
        for i, batch in zip(t, batches):
            if not args.use_threaded_data_workers:
                batch = train_dataset[batch]
            (source, target, label, subword_mask, unique_sources_indices,
             unique_sources_counts, unique_targets_indices,
             unique_targets_counts) = batch
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
                    context[0]), emb_in_grad_normalization, with_sparsity=True,
                last_update_buffer=last_update_buffer,
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
                # Embedding in
                embedding_in_norm = embedding_in.weight.data(
                    ctx=context[0]).as_in_context(
                        mx.cpu()).tostype("default").norm(axis=1)
                sw.add_histogram(tag='embedding_in_word_norm',
                                 values=embedding_in_norm[:len(vocab)],
                                 global_step=current_update, bins=200)
                sw.add_histogram(tag='embedding_in_subword_norm',
                                 values=embedding_in_norm[:len(vocab)],
                                 global_step=current_update, bins=200)
                embedding_in_grad = embedding_in.weight.grad(
                    ctx=context[0]).as_in_context(
                        mx.cpu()).tostype("default").norm(axis=1)
                sw.add_histogram(tag='embedding_in_word_grad',
                                 values=embedding_in_grad[:len(vocab)],
                                 global_step=current_update, bins=200)
                sw.add_histogram(tag='embedding_in_subword_grad',
                                 values=embedding_in_grad[len(vocab):],
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

                eval_dict = evaluation.evaluate(args, embedding_in, None,
                                                vocab, subword_vocab, sw)
                for k, v in eval_dict.items():
                    sw.add_scalar(tag=k, value=float(v),
                                  global_step=current_update)

                sw.flush()

                # Save params after evaluation
                utils.save_params(args, embedding_in, embedding_out, None,
                                  current_update)

        # Shut down data preloading executor
        if args.use_threaded_data_workers:
            executor.shutdown()
