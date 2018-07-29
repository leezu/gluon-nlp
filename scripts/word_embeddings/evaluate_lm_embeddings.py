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
# pylint: disable=logging-too-many-args
"""Evaluation of pretrained word embeddings of a language model
===============================================================

This example shows how to load and perform intrinsic evaluation of word
embeddings using a variety of datasets all part of the Gluon NLP Toolkit.

"""

import argparse
import logging
import os
import sys

import evaluation
import gluonnlp as nlp
import utils


def get_args():
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        description='Word embedding evaluation with Gluon.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Embeddings arguments
    group = parser.add_argument_group('Embedding arguments')
    group.add_argument(
        '--model', type=str,
        help='Name of a pretrained model included in gluonnlp.model')
    group.add_argument(
        '--max-vocab-size', type=int, default=None,
        help=('Only retain the X first tokens from the pretrained embedding. '
              'The tokens are ordererd by decreasing frequency.'
              'As the analogy task takes the whole vocabulary into account, '
              'removing very infrequent words improves performance.'))

    # Computation options
    group = parser.add_argument_group('Computation arguments')
    group.add_argument(
        '--batch-size', type=int, default=1024,
        help='Batch size to use on analogy task. '
        'Decrease batch size if evaluation crashes.')
    group.add_argument(
        '--gpu', type=int, help=('Number (index) of GPU to run on, e.g. 0. '
                                 'If not specified, uses CPU.'))
    group.add_argument('--no-hybridize', action='store_true',
                       help='Disable hybridization of gluon HybridBlocks.')

    # Logging
    group = parser.add_argument_group('Logging arguments')
    group.add_argument('--logdir', type=str, default='logs',
                       help='Directory to store logs.')

    # Evaluation options
    evaluation.add_parameters(parser)

    args = parser.parse_args()

    validate_args(args)
    evaluation.validate_args(args)

    return args


def validate_args(args):
    """Validate provided arguments"""
    supported_models = [
        'awd_lstm_lm_1150', 'awd_lstm_lm_600', 'standard_lstm_lm_200',
        'standard_lstm_lm_650', 'standard_lstm_lm_1500', 'big_rnn_lm_2048_512'
    ]
    if args.model not in supported_models:
        print('--model "{}" is not valid.'.format(args.model))
        print('Valid choices are', ", ".join(supported_models))
        sys.exit(1)
    print(args)


def _load_awd(args):
    if "600" in args.model:
        model, vocab = nlp.model.awd_lstm_lm_600(dataset_name='wikitext-2',
                                                 pretrained=True)
    elif "1150" in args.model:
        model, vocab = nlp.model.awd_lstm_lm_1150(dataset_name='wikitext-2',
                                                  pretrained=True)
    else:
        raise RuntimeError('Invalid model specified: {}'.format(args.model))
    token_embedding = nlp.embedding.TokenEmbedding(
        unknown_token=vocab.unknown_token, allow_extend=True)
    token_embedding[vocab.idx_to_token] = model.embedding[0].weight.data()
    return token_embedding


def _load_standard_lstm(args):
    if "200" in args.model:
        model, vocab = nlp.model.standard_lstm_lm_200(
            dataset_name='wikitext-2', pretrained=True)
    elif "650" in args.model:
        model, vocab = nlp.model.standard_lstm_lm_650(
            dataset_name='wikitext-2', pretrained=True)
    elif "1500" in args.model:
        model, vocab = nlp.model.standard_lstm_lm_1500(
            dataset_name='wikitext-2', pretrained=True)
    else:
        raise RuntimeError('Invalid model specified: {}'.format(args.model))
    token_embedding = nlp.embedding.TokenEmbedding(
        unknown_token=vocab.unknown_token, allow_extend=True)
    token_embedding[vocab.idx_to_token] = model.embedding[0].weight.data()
    return token_embedding


def _load_bigrnn(args):
    model, vocab = nlp.model.big_rnn_lm_2048_512(dataset_name='gbw',
                                                 pretrained=True)
    token_embedding = nlp.embedding.TokenEmbedding(
        unknown_token=vocab.unknown_token, allow_extend=True)
    token_embedding[vocab.idx_to_token] = model.embedding[0].weight.data()
    return token_embedding


def load_embedding(args):
    """Load a TokenEmbedding from a pretrained language model."""
    if 'awd' in args.model:
        return _load_awd(args)
    elif 'standard' in args.model:
        return _load_standard_lstm(args)
    elif 'big' in args.model:
        return _load_bigrnn(args)
    else:
        raise RuntimeError('Invalid model specified: {}'.format(args.model))


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    args_ = get_args()
    ctx = utils.get_context(args_)[0]
    os.makedirs(args_.logdir, exist_ok=True)

    # Load pretrained embeddings
    token_embedding = load_embedding(args_)
    known_tokens = set(token_embedding.idx_to_token)

    if args_.max_vocab_size:
        size = min(len(token_embedding._idx_to_token), args_.max_vocab_size)
        token_embedding._idx_to_token = token_embedding._idx_to_token[:size]
        token_embedding._idx_to_vec = token_embedding._idx_to_vec[:size]
        token_embedding._token_to_idx = {
            token: idx
            for idx, token in enumerate(token_embedding._idx_to_token)
        }

    similarity_results = evaluation.evaluate_similarity(
        args_, token_embedding, ctx, known_tokens, logfile=os.path.join(
            args_.logdir, 'similarity.tsv'))
    analogy_results = evaluation.evaluate_analogy(
        args_, token_embedding, ctx, known_tokens, logfile=os.path.join(
            args_.logdir, 'analogy.tsv'))
