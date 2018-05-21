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

import collections
import logging

from mxboard import SummaryWriter
from mxnet import gluon

import arguments
import evaluation
import gluonnlp as nlp
import utils


def add_parameters(parser):
    group = parser.add_argument_group('Evaluation only specific settings')
    group.add_argument('path', type=str,
                       help='Path to pretrained TokenEmbedding file.')


def get_model(args):
    context = utils.get_context(args)
    embeddings_context = [context[0]]

    token_embedding = nlp.embedding.TokenEmbedding.from_file(args.path)
    assert args.emsize == token_embedding.idx_to_vec.shape[1]
    vocab = nlp.Vocab(
        collections.Counter(token_embedding.idx_to_token), unknown_token=None,
        padding_token=None, bos_token=None, eos_token=None)
    vocab.set_embedding(token_embedding)

    # Output embeddings
    embedding = gluon.nn.SparseEmbedding(
        len(token_embedding.idx_to_token), args.emsize)
    embedding.initialize(ctx=embeddings_context)
    embedding.weight.set_data(vocab.embedding.idx_to_vec)

    return vocab, embedding


def load_and_evaluate(args):
    vocab, embedding_in = get_model(args)
    sw = SummaryWriter(logdir=args.logdir)
    eval_dict = evaluation.evaluate(args, embedding_in, None, None, vocab,
                                    None, sw)
    for k, v in eval_dict.items():
        for i in range(100):
            sw.add_scalar(tag=k, value=float(v), global_step=i)
    for k, v in eval_dict.items():
        print('{:.2f}'.format(v), '\t', k)


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    args_ = arguments.get_and_setup([add_parameters])
    load_and_evaluate(args_)
