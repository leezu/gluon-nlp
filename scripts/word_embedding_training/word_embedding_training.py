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
import tqdm
from mxboard import SummaryWriter
from mxnet import gluon

import arguments
import data
import evaluation
import model
import utils
import trainer


###############################################################################
# Training code
###############################################################################
def log(args, sw, embedding_in, embedding_out, subword_net, embedding_net,
        auxilary_task_net, loss, task_loss, aux_loss, aux_acc,
        attention_regularization, subword_embeddings, num_update, vocab,
        subword_vocab):
    context = utils.get_context(args)
    # Mxboard
    if embedding_in is not None:
        embedding_in_norm = embedding_in.weight.data(
            ctx=context[0]).as_in_context(
                mx.cpu()).tostype("default").norm(axis=1)
        sw.add_histogram(tag='embedding_in_norm', values=embedding_in_norm,
                         global_step=num_update, bins=200)
        embedding_in_grad_norm = embedding_in.weight.grad(
            ctx=context[0]).as_in_context(
                mx.cpu()).tostype("default").norm(axis=1)
        sw.add_histogram(tag='embedding_in_grad_norm',
                         values=embedding_in_grad_norm, global_step=num_update,
                         bins=200)
        embedding_in_data = embedding_in.weight.data(
            ctx=context[0]).as_in_context(mx.cpu()).tostype("default")
        sw.add_histogram(tag='embedding_in', values=embedding_in_data,
                         global_step=num_update, bins=200)
        embedding_in_grad = embedding_in.weight.grad(
            ctx=context[0]).as_in_context(mx.cpu()).tostype("default")
        sw.add_histogram(tag='embedding_in_grad', values=embedding_in_grad,
                         global_step=num_update, bins=200)
        # Learning rate
    if subword_net is not None:
        for k, v in subword_net.collect_params().items():
            sw.add_histogram(tag=k, values=v.data(ctx=context[0]),
                             global_step=num_update, bins=200)
            if v.grad_req != 'null':
                sw.add_histogram(tag='grad-' + str(k),
                                 values=v.grad(ctx=context[0]),
                                 global_step=num_update, bins=200)

        # Also report norm of the subword embeddings
        if args.subword_network.lower() in ['fasttext', 'sumreduce']:
            subword_embedding_norm = embedding_in.weight.data(
                ctx=context[0]).as_in_context(
                    mx.cpu()).tostype("default").norm(axis=1)
            sw.add_histogram(tag='subword_embedding_norm',
                             values=subword_embedding_norm,
                             global_step=num_update, bins=200)
            subword_embedding_grad_norm = embedding_in.weight.grad(
                ctx=context[0]).as_in_context(
                    mx.cpu()).tostype("default").norm(axis=1)
            sw.add_histogram(tag='subword_embedding_grad_norm',
                             values=subword_embedding_grad_norm,
                             global_step=num_update, bins=200)

        # Log the embeddings predicted for this batch based on the subword
        # units
        if subword_embeddings is not None:
            sw.add_histogram(tag='predicted_embedding_in_norm',
                             values=subword_embeddings.norm(axis=1),
                             global_step=num_update, bins=200)

    if embedding_net is not None:
        for k, v in embedding_net.collect_params().items():
            if v.grad_req == 'null':
                continue
            sw.add_histogram(tag=k, values=v.data(ctx=context[0]),
                             global_step=num_update, bins=200)
            sw.add_histogram(tag='grad-' + str(k),
                             values=v.grad(ctx=context[0]),
                             global_step=num_update, bins=200)

    if auxilary_task_net is not None:
        for k, v in auxilary_task_net.collect_params().items():
            if v.grad_req == 'null':
                continue
            sw.add_histogram(tag=k, values=v.data(ctx=context[0]),
                             global_step=num_update, bins=200)
            sw.add_histogram(tag='grad-' + str(k),
                             values=v.grad(ctx=context[0]),
                             global_step=num_update, bins=200)

    # Embedding out
    if embedding_out is not None:
        embedding_out_norm = embedding_out.weight.data(
            ctx=context[0]).as_in_context(
                mx.cpu()).tostype("default").norm(axis=1)
        sw.add_histogram(tag='embedding_out_norm', values=embedding_out_norm,
                         global_step=num_update, bins=200)
        embedding_out_grad_norm = embedding_out.weight.grad(
            ctx=context[0]).as_in_context(
                mx.cpu()).tostype("default").norm(axis=1)
        sw.add_histogram(tag='embedding_out_grad_norm',
                         values=embedding_out_grad_norm,
                         global_step=num_update, bins=200)
        embedding_out_data = embedding_out.weight.data(
            ctx=context[0]).as_in_context(mx.cpu()).tostype("default")
        sw.add_histogram(tag='embedding_out', values=embedding_out_data,
                         global_step=num_update, bins=200)
        embedding_out_grad = embedding_out.weight.grad(
            ctx=context[0]).as_in_context(mx.cpu()).tostype("default")
        sw.add_histogram(tag='embedding_out_grad', values=embedding_out_grad,
                         global_step=num_update, bins=200)

    # Scalars
    if not isinstance(loss, int):
        sw.add_scalar(tag='loss', value=loss.mean().asscalar(),
                      global_step=num_update)
    if not isinstance(task_loss, int):
        sw.add_scalar(tag='task_loss', value=task_loss.asscalar(),
                      global_step=num_update)
    if not isinstance(aux_loss, int):
        sw.add_scalar(tag='aux_loss', value=aux_loss.asscalar(),
                      global_step=num_update)
    if not isinstance(aux_acc, int):
        sw.add_scalar(tag='aux_acc', value=aux_acc.asscalar(),
                      global_step=num_update)
    if not isinstance(attention_regularization, int):
        sw.add_scalar(tag='attention_regularization',
                      value=attention_regularization.asscalar(),
                      global_step=num_update)

    eval_dict = evaluation.evaluate(args, embedding_in, subword_net,
                                    embedding_net, vocab, subword_vocab, sw)
    for k, v in eval_dict.items():
        sw.add_scalar(tag=k, value=float(v), global_step=num_update)

    sw.flush()


def log_learning_rates(sw, num_update, embedding_in_trainer,
                       embedding_out_trainer, subword_trainer):
    if embedding_in_trainer is not None:
        sw.add_scalar(tag='embedding_in_lr',
                      value=embedding_in_trainer.learning_rate,
                      global_step=num_update)
    if subword_trainer is not None:
        sw.add_scalar(tag='subword_net_lr',
                      value=subword_trainer.learning_rate,
                      global_step=num_update)
    if embedding_out_trainer is not None:
        sw.add_scalar(tag='embedding_out_lr',
                      value=embedding_out_trainer.learning_rate,
                      global_step=num_update)


###############################################################################
# Training code
###############################################################################
def compute_subword_embeddings(args, *posargs, **kwargs):
    context = utils.get_context(args)

    if len(context) == 1:
        # Simplified single GPU code
        return _compute_subword_embeddings_single_gpu(args, *posargs, **kwargs)
    else:
        return _compute_subword_embeddings(args, *posargs, **kwargs)


def compute_embedding_net_self_attention(args, embedding_net, encoded,
                                         mask_ctx):
    out, att_weights = embedding_net(encoded, mask_ctx)
    attention_regularizer = mx.nd.sqrt(
        mx.nd.sum((mx.nd.batch_dot(att_weights, att_weights.swapaxes(1, 2)) -
                   mx.nd.eye(args.self_attention_num_attention,
                             ctx=att_weights.context))**2))
    return out, attention_regularizer


def compute_auxilary_task(args, auxilary_task_net, aux_loss_function, encoded,
                          word_indices_ctx, last_valid_ctx):
    aux_input = encoded[(last_valid_ctx,
                         mx.nd.arange(encoded.shape[1], ctx=encoded.context))]
    aux_pred = auxilary_task_net(aux_input)
    aux_loss = mx.nd.sum(aux_loss_function(aux_pred, word_indices_ctx))
    aux_acc = aux_pred.argmax(1) == word_indices_ctx
    aux_acc = mx.nd.sum(aux_acc) / aux_acc.shape[0]
    return aux_loss, aux_acc


def _compute_subword_embeddings_single_gpu(
        args, subword_net, embedding_net, auxilary_task_net, aux_loss_function,
        source_subword, unique_sources_indices,
        unique_sources_subwordsequences, unique_sources_subwordsequences_mask,
        unique_sources_subwordsequences_last_valid):
    context = utils.get_context(args)
    assert len(context) == 1

    attention_regularization = 0
    aux_loss = 0

    word_indices_ctx = unique_sources_indices[0]
    subwordsequences_ctx = unique_sources_subwordsequences[0]
    mask_ctx = unique_sources_subwordsequences_mask[0]
    last_valid_ctx = unique_sources_subwordsequences_last_valid[0]

    encoded = subword_net(subwordsequences_ctx, mask_ctx)

    # Compute embedding from encoded subword sequence
    if embedding_net is not None:
        if (args.embedding_network.lower() == 'selfattentionembedding'):
            subword_embedding_weights, attention_regularization = \
                compute_embedding_net_self_attention(
                    args, embedding_net, encoded, mask_ctx)
        else:
            subword_embedding_weights = embedding_net(encoded, last_valid_ctx)
            attention_regularization = 0
    else:
        subword_embedding_weights = encoded
        attention_regularization = 0

    # Auxilary task
    if auxilary_task_net is not None:
        aux_loss, aux_acc = compute_auxilary_task(
            args, auxilary_task_net, aux_loss_function, encoded,
            word_indices_ctx, last_valid_ctx)
    else:
        aux_loss = 0
        aux_acc = 0

    # Look up subword embeddings of batch
    subword_embeddings = mx.nd.Embedding(
        data=source_subword, weight=subword_embedding_weights,
        input_dim=subword_embedding_weights.shape[0], output_dim=args.emsize)

    return subword_embeddings, aux_loss, aux_acc, attention_regularization


def _compute_subword_embeddings(
        args, subword_net, embedding_net, auxilary_task_net, aux_loss_function,
        source_subword, unique_sources_indices,
        unique_sources_subwordsequences, unique_sources_subwordsequences_mask,
        unique_sources_subwordsequences_last_valid):
    context = utils.get_context(args)
    # Compute subword embeddings from subword info (byte sequences)
    subword_embedding_weights = []
    attention_regularization = 0
    aux_loss = 0
    aux_acc = 0
    for (word_indices_ctx,
         subwordsequences_ctx, mask_ctx, last_valid_ctx) in zip(
             unique_sources_indices, unique_sources_subwordsequences,
             unique_sources_subwordsequences_mask,
             unique_sources_subwordsequences_last_valid):
        encoded = subword_net(subwordsequences_ctx, mask_ctx)

        # Compute embedding from encoded subword sequence
        if (args.embedding_network.lower() == 'selfattentionembedding'):
            subword_embedding_weights_ctx, attention_regularization_ctx = \
                compute_embedding_net_self_attention(
                    args, embedding_net, encoded, mask_ctx)
            attention_regularization = \
                attention_regularization + \
                attention_regularization_ctx.as_in_context(context[0])
        else:
            subword_embedding_weights_ctx = embedding_net(
                encoded, last_valid_ctx)
        subword_embedding_weights.append(
            subword_embedding_weights_ctx.as_in_context(context[0]))

        # Auxilary task
        if auxilary_task_net is not None:
            aux_loss_ctx, aux_acc_ctx = compute_auxilary_task(
                args, auxilary_task_net, aux_loss_function, encoded,
                word_indices_ctx, last_valid_ctx)
            aux_loss = aux_loss + aux_loss_ctx.as_in_context(context[0])
            aux_acc = aux_acc + aux_acc_ctx.as_in_context(context[0])

    subword_embedding_weights = mx.nd.concat(*subword_embedding_weights, dim=0)

    # Look up subword embeddings of batch
    subword_embeddings = mx.nd.Embedding(
        data=source_subword, weight=subword_embedding_weights,
        input_dim=subword_embedding_weights.shape[0], output_dim=args.emsize)

    return subword_embeddings, aux_loss, aux_acc, attention_regularization


def compute_word_embeddings(args, embedding_in, subword_embeddings, source):
    # Sum word and subword level embeddings
    if embedding_in is not None:
        word_embeddings = embedding_in(source)
        if subword_embeddings is not None:
            emb_in = subword_embeddings + word_embeddings
        else:
            emb_in = word_embeddings
        return emb_in

    # Word embedding matrix is disabled
    else:
        return subword_embeddings


def train(args):
    utils.warnings_and_asserst(args)

    train_dataset, vocab, subword_vocab = data.get_train_data(args)
    (embedding_in, embedding_out, subword_net, embedding_net,
     auxilary_task_net, loss_function, aux_loss_function) = model.get_model(
         args, train_dataset, vocab, subword_vocab)
    context = utils.get_context(args)

    embedding_out_trainer = trainer.get_embedding_out_trainer(
        args, embedding_out.collect_params())
    embedding_in_trainer = None
    subword_trainer = None
    if embedding_in is not None:
        embedding_in_trainer = trainer.get_embedding_in_trainer(
            args, embedding_in.collect_params(), len(vocab))
    if subword_net is not None:
        subword_params = list(subword_net.collect_params().values())
        if embedding_net is not None:
            subword_params += list(embedding_net.collect_params().values())
        if auxilary_task_net is not None:
            subword_params += list(auxilary_task_net.collect_params().values())
        subword_trainer = trainer.get_subword_trainer(args, subword_params,
                                                      len(subword_vocab))

    # Logging writer
    sw = SummaryWriter(logdir=args.logdir)
    sw.add_text(tag='args', text=str(args), global_step=0)

    indices = np.arange(len(train_dataset))
    num_update = 0
    for epoch in range(args.epochs):
        np.random.shuffle(indices)
        with utils.print_time('create batch indices'):
            batches = [
                indices[i:i + args.batch_size]
                for i in range(0, len(indices), args.batch_size)
            ]

        for i, batch_idx in tqdm.tqdm(
                enumerate(batches), total=len(batches), ascii=True,
                smoothing=1, dynamic_ncols=True):
            num_update += len(batch_idx)
            if 'rnn' in args.subword_network.lower():
                mx.nd.waitall()  # wait to avoid cudnn memory related crashes

            progress = (epoch * len(batches) + i) / (
                args.epochs * len(batches))
            batch = train_dataset[batch_idx]
            (source_np, target, label, unique_sources_indices_np,
             unique_sources_counts, unique_sources_subwordsequences,
             source_subword, unique_sources_subwordsequences_mask,
             unique_sources_subwordsequences_count_nonmasked,
             unique_sources_unique_subwordsequences_indices,
             unique_sources_unique_subwordsequences_counts,
             unique_targets_indices, unique_targets_counts) = batch

            unique_sources_subwordsequences_last_valid = \
                (unique_sources_subwordsequences_mask == 0).argmax(axis=1) - 1
            unique_sources_subwordsequences_last_valid[
                unique_sources_subwordsequences_last_valid ==
                -1] = unique_sources_subwordsequences_mask.shape[1] - 1

            # Load data for training embedding matrix to context[0]
            source = mx.nd.array(source_np, ctx=context[0])
            target = mx.nd.array(target, ctx=context[0])
            label = mx.nd.array(label, ctx=context[0])

            # Load indices for looking up subword embedding to context[0]
            source_subword = mx.nd.array(source_subword, ctx=context[0])

            # Split and load subword info to all GPUs for accelerated computation
            assert unique_sources_subwordsequences.shape == \
                unique_sources_subwordsequences_mask.shape
            num_unique_subwordsequences = \
                unique_sources_subwordsequences.shape[0]
            unique_sources_indices_split = gluon.utils.split_and_load(
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
                # Compute subword level embeddings
                if subword_net is not None:
                    (subword_embeddings, aux_loss, aux_acc,
                     attention_regularization) = compute_subword_embeddings(
                         args, subword_net, embedding_net, auxilary_task_net,
                         aux_loss_function, source_subword,
                         unique_sources_indices_split,
                         unique_sources_subwordsequences,
                         unique_sources_subwordsequences_mask,
                         unique_sources_subwordsequences_last_valid)
                else:
                    subword_embeddings = None
                    aux_loss = 0
                    aux_acc = 0
                    attention_regularization = 0

                # Combine subword level embeddings with word embeddings
                emb_in = compute_word_embeddings(args, embedding_in,
                                                 subword_embeddings, source)

                # Target embeddings
                emb_out = embedding_out(target)

                # Compute loss
                pred = mx.nd.batch_dot(emb_in, emb_out.swapaxes(1, 2))
                task_loss = loss_function(pred, label)

                # Normalize task loss
                if 'count' in args.normalize_loss:
                    _, source_count_inverse, source_count = np.unique(
                        source_np, return_counts=True, return_inverse=True)
                    if 'log' in args.normalize_loss:
                        source_count = np.log(source_count) + 1
                    task_loss_normalization = 1 / mx.nd.array(
                        source_count[source_count_inverse], ctx=context[0])
                    normalized_task_loss = mx.nd.dot(task_loss_normalization,
                                                     task_loss)
                elif args.normalize_loss == 'mean':
                    normalized_task_loss = mx.nd.mean(task_loss)
                elif args.normalize_loss == 'none':
                    normalized_task_loss = mx.nd.sum(task_loss)
                else:
                    logging.error('--normalize-loss {} is invalid'.format(
                        args.normalize_loss))
                    sys.exit(1)

                loss =  normalized_task_loss + aux_loss + \
                    attention_regularization

            loss.backward()

            # Update parameters
            if subword_net is not None:
                if args.subword_network.lower() in ['fasttext', 'sumreduce']:
                    subword_trainer.set_learning_rate(args.subword_sparse_l2 *
                                                      (1 - progress))
                else:
                    subword_trainer.set_learning_rate(args.subword_dense_l2 *
                                                      (1 - progress))
                if args.normalize_gradient.lower() in ['count', 'l2']:
                    assert args.subword_network.lower() in \
                        ['fasttext', 'sumreduce'], 'Normalizing dense grad ' \
                        ' by count or L2 is not supported.'

                    trainer.normalize_sparse_grads(
                        args, subword_net.embedding,
                        unique_sources_unique_subwordsequences_counts,
                        unique_sources_unique_subwordsequences_indices)
                elif args.normalize_gradient.lower() == 'batch_size':
                    if args.subword_network.lower() in \
                             ['fasttext', 'sumreduce']:
                        subword_trainer.step(
                            batch_size=
                            unique_sources_subwordsequences_count_nonmasked)
                    else:
                        subword_trainer.step(
                            batch_size=num_unique_subwordsequences)
                elif args.normalize_gradient.lower() == 'none':
                    subword_trainer.step(batch_size=1)
                else:
                    logging.error('--normalize-gradient {} is invalid'.format(
                        args.normalize_gradient))
                    sys.exit(1)

            if embedding_in is not None:
                embedding_in_trainer.set_learning_rate(args.word_l2 *
                                                       (1 - progress))
                # Force eager update before evaluation
                if i % args.eval_interval == 0:
                    embedding_in_trainer.lazy_update = False
                if args.normalize_gradient.lower() in ['count', 'l2']:
                    trainer.normalize_sparse_grads(args, embedding_in,
                                                   unique_sources_counts,
                                                   unique_sources_indices_np)
                    embedding_in_trainer.step(batch_size=1)
                elif args.normalize_gradient.lower() == 'batch_size':
                    embedding_in_trainer.step(
                        batch_size=source.shape[0] * source.shape[1])
                elif args.normalize_gradient.lower() == 'none':
                    embedding_in_trainer.step(batch_size=1)
                else:
                    logging.error('--normalize-gradient {} is invalid'.format(
                        args.normalize_gradient))
                    sys.exit(1)

                embedding_in_trainer.lazy_update = True

            embedding_out_trainer.set_learning_rate(args.word_l2 *
                                                    (1 - progress))
            if args.normalize_gradient.lower() in ['count', 'l2']:
                trainer.normalize_sparse_grads(args, embedding_out,
                                               unique_targets_counts,
                                               unique_targets_indices)
                embedding_out_trainer.step(batch_size=1)
            elif args.normalize_gradient.lower() == 'batch_size':
                embedding_out_trainer.step(
                    batch_size=target.shape[0] * target.shape[1])
            elif args.normalize_gradient.lower() == 'none':
                embedding_out_trainer.step(batch_size=1)
            else:
                logging.error('--normalize-gradient {} is invalid'.format(
                    args.normalize_gradient))
                sys.exit(1)

            # Logging
            if i % args.eval_interval == 0:
                with utils.print_time('mx.nd.waitall()'):
                    mx.nd.waitall()

                log(args, sw, embedding_in, embedding_out, subword_net,
                    embedding_net, auxilary_task_net, loss,
                    normalized_task_loss, aux_loss, aux_acc,
                    attention_regularization, subword_embeddings, num_update,
                    vocab, subword_vocab)
                log_learning_rates(sw, num_update, embedding_in_trainer,
                                   embedding_out_trainer, subword_trainer)

                # Save params after evaluation
                utils.save_params(args, embedding_in, embedding_out,
                                  subword_net, num_update)

    sw.close()


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    args_ = arguments.get_and_setup()
    train(args_)
