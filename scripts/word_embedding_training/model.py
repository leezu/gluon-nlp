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

import mxnet as mx
from mxnet import gluon

import subword
import utils


def add_parameters(parser):
    group = parser.add_argument_group('Model arguments')
    group.add_argument('--no-token-embedding', action='store_true',
                       help='Don\'t use any token embedding. '
                       'Only use subword units.')
    group.add_argument(
        '--subword-network', type=str, default='', nargs='?',
        help='Network architecture to encode subword level information. ' +
        str(subword.list_subwordnetworks()))
    group.add_argument(
        '--embedding-network', default='LastOutputEmbedding',
        help='Network architecture to create embedding from encoded subwords.')
    group.add_argument('--auxilary-task', action='store_true',
                       help='Use auxilary word prediction task.')
    group.add_argument('--objective', type=str, default='skipgram',
                       help='Word embedding training objective.')
    group.add_argument('--emsize', type=int, default=300,
                       help='Size of word embeddings')
    group.add_argument(
        '--no-normalized-initialization', action='store_true',
        help='Normalize uniform initialization range by embedding size.')
    group.add_argument(
        '--no-zero-embedding-out-initialization', action='store_true',
        help='Don\'t initialize context embedding matrix as 0 as in fasttext.')


def get_model(args, train_dataset, vocab, subword_vocab):
    # Must use at least one of word level or subword level embeddings
    assert not (args.no_token_embedding and not args.subword_network)

    num_tokens = train_dataset.num_tokens
    context = utils.get_context(args)
    embeddings_context = [context[0]]

    if not args.no_normalized_initialization:
        embedding_initializer = mx.init.Uniform(scale=1 / args.emsize)
    else:
        embedding_initializer = mx.init.Uniform()
    if not args.no_zero_embedding_out_initialization:
        embedding_out_initializer = mx.init.Zero()
    else:
        embedding_out_initializer = embedding_initializer

    # Output embeddings
    embedding_out = gluon.nn.SparseEmbedding(
        num_tokens, args.emsize, weight_initializer=embedding_out_initializer)
    embedding_out.initialize(ctx=embeddings_context)
    if not args.dont_hybridize:
        embedding_out.hybridize()

    # Word level input embeddings
    if not args.no_token_embedding:
        embedding_in = gluon.nn.SparseEmbedding(
            num_tokens, args.emsize, weight_initializer=embedding_initializer)
        embedding_in.initialize(ctx=embeddings_context)
        if not args.dont_hybridize:
            embedding_in.hybridize()
    else:
        embedding_in = None

    # Subword level input embeddings
    if args.subword_network:
        # Fasttext or RNN
        if args.subword_network.lower() in ['fasttext', 'sumreduce']:
            # Fasttext mode
            subword_net = subword.create(
                name=args.subword_network, args=args,
                vocab_size=len(subword_vocab),
                embedding_initializer=embedding_initializer)
            embedding_net = None
            auxilary_task_net = None
        else:
            # RNN mode
            subword_net = subword.create(name=args.subword_network, args=args,
                                         vocab_size=len(subword_vocab))
            embedding_net = subword.create(name=args.embedding_network,
                                           args=args)
            embedding_net.initialize(mx.init.Orthogonal(), ctx=context)

            if not args.dont_hybridize:
                embedding_net.hybridize()

            if args.auxilary_task:
                auxilary_task_net = subword.create(
                    name='WordPrediction', vocab_size=len(vocab), args=args)
                auxilary_task_net.initialize(mx.init.Orthogonal(), ctx=context)
                if not args.dont_hybridize:
                    auxilary_task_net
            else:
                auxilary_task_net = None

        subword_net.initialize(mx.init.Xavier(), ctx=context)
        if not args.dont_hybridize:
            subword_net.hybridize()
    else:
        subword_net = None
        embedding_net = None
        auxilary_task_net = None

    loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
    aux_loss = gluon.loss.SoftmaxCrossEntropyLoss()

    return (embedding_in, embedding_out, subword_net, embedding_net,
            auxilary_task_net, loss, aux_loss)
