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

import mxnet as mx
import numpy as np
from mxboard import SummaryWriter
from mxnet import gluon

import arguments
import data
import evaluation
import fasttext
import subword
import utils

try:
    import tqdm
except ImportError:
    logging.warning('tqdm not installed. '
                    ' Install via `pip install tqdm` for better usability.')
    tqdm = None


###############################################################################
# Build the model
###############################################################################
def get_model(args, train_dataset, vocab, subword_vocab):
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
        embedding_net = subword.create(name=args.embedding_network, args=args)
        embedding_net.initialize(mx.init.Orthogonal(), ctx=context)

        if not args.dont_hybridize:
            subword_net.hybridize()
            embedding_net.hybridize()

        if args.auxilary_task:
            auxilary_task_net = subword.create(
                name='WordPrediction', vocab_size=len(vocab), args=args)
            auxilary_task_net.initialize(mx.init.Orthogonal(), ctx=context)
            if not args.dont_hybridize:
                auxilary_task_net
        else:
            auxilary_task_net = None

    else:
        subword_net = None
        embedding_net = None

    loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
    aux_loss = gluon.loss.SoftmaxCrossEntropyLoss()

    return (embedding_in, embedding_out, subword_net, embedding_net,
            auxilary_task_net, loss, aux_loss)


###############################################################################
# Training code
###############################################################################
def train(args):
    train_dataset, vocab, subword_vocab = data.get_train_data(args)
    (embedding_in, embedding_out, subword_net, embedding_net,
     auxilary_task_net, loss_function, aux_loss_function) = get_model(
         args, train_dataset, vocab, subword_vocab)
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
            dense_params = (list(subword_net.collect_params().values()) +
                            list(embedding_net.collect_params().values()))
            if auxilary_task_net is not None:
                dense_params = (dense_params + list(
                    auxilary_task_net.collect_params().values()))
            dense_trainer = gluon.Trainer(dense_params, args.dense_optimizer, {
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
            mx.nd.waitall()  # wait to avoid cudnn memory related crashes

            batch = train_dataset[batch_idx]
            (source, target, label, unique_sources_indices_np,
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
            unique_sources_indices = gluon.utils.split_and_load(
                unique_sources_indices_np, context, even_split=False)
            unique_sources_subwordsequences = gluon.utils.split_and_load(
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
                    attention_regularization = 0
                    aux_loss = 0
                    for (word_indices_ctx, subwordsequences_ctx,
                         mask_ctx, last_valid_ctx) in zip(
                             unique_sources_indices,
                             unique_sources_subwordsequences,
                             unique_sources_subwordsequences_mask,
                             unique_sources_subwordsequences_last_valid):
                        encoded, states = subword_net(subwordsequences_ctx,
                                                      mask_ctx)

                        # Compute embedding from encoded subword sequence
                        if (args.embedding_network.lower() ==
                                'selfattentionembedding'):
                            out, att_weights = embedding_net(encoded, mask_ctx)
                            attention_regularizer = mx.nd.sqrt(
                                mx.nd.sum((mx.nd.batch_dot(
                                    att_weights, att_weights.swapaxes(
                                        1, 2)) - mx.nd.eye(
                                            args.self_attention_num_attention,
                                            ctx=att_weights.context))**2))
                            attention_regularization = (
                                attention_regularization +
                                attention_regularizer.as_in_context(
                                    context[0]))
                        else:
                            out = embedding_net(encoded, last_valid_ctx)

                        # Auxilary task
                        if auxilary_task_net is not None:
                            aux_input = encoded[(last_valid_ctx,
                                                 mx.nd.arange(
                                                     encoded.shape[1],
                                                     ctx=encoded.context))]
                            aux_pred = auxilary_task_net(aux_input)
                            aux_loss = aux_loss + mx.nd.sum(
                                aux_loss_function(
                                    aux_pred, word_indices_ctx).as_in_context(
                                        context[0]))
                            aux_acc = aux_pred.argmax(1) == word_indices_ctx
                            aux_acc = mx.nd.sum(aux_acc) / aux_acc.shape[0]
                        else:
                            # TODO aux_acc only computed on one device
                            aux_acc = 0

                        # TODO check if the gradient is actually passed when switching device
                        subword_embedding_weights.append(
                            out.as_in_context(context[0]))
                    subword_embedding_weights = mx.nd.concat(
                        *subword_embedding_weights, dim=0)

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
                task_loss = mx.nd.sum(loss_function(pred, label))
                loss = task_loss + aux_loss + attention_regularization

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
                        (-1, 1)), unique_sources_indices_np), ctx=context[0],
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
                        if v.grad_req == 'null':
                            continue
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

                if embedding_net is not None:
                    for k, v in embedding_net.collect_params().items():
                        if v.grad_req == 'null':
                            continue
                        sw.add_histogram(tag=k, values=v.data(ctx=context[0]),
                                         global_step=current_update, bins=200)
                        sw.add_histogram(tag='grad-' + str(k),
                                         values=v.grad(ctx=context[0]),
                                         global_step=current_update, bins=200)

                if auxilary_task_net is not None:
                    for k, v in auxilary_task_net.collect_params().items():
                        if v.grad_req == 'null':
                            continue
                        sw.add_histogram(tag=k, values=v.data(ctx=context[0]),
                                         global_step=current_update, bins=200)
                        sw.add_histogram(tag='grad-' + str(k),
                                         values=v.grad(ctx=context[0]),
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
                sw.add_scalar(tag='task_loss',
                              value=task_loss.mean().asscalar(),
                              global_step=current_update)
                if not isinstance(aux_loss, int):
                    sw.add_scalar(tag='aux_loss', value=aux_loss.asscalar(),
                                  global_step=current_update)
                if not isinstance(aux_acc, int):
                    sw.add_scalar(tag='aux_acc', value=aux_acc.asscalar(),
                                  global_step=current_update)
                if not isinstance(attention_regularization, int):
                    sw.add_scalar(tag='attention_regularization',
                                  value=attention_regularization.asscalar(),
                                  global_step=current_update)

                eval_dict = evaluation.evaluate(args, embedding_in,
                                                subword_net, embedding_net,
                                                vocab, subword_vocab, sw)
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

    args_ = arguments.get_and_setup()

    if not args_.subword_network == 'fasttext':
        train(args_)
    else:
        fasttext.train(args_)
