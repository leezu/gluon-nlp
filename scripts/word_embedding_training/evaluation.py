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

import itertools
import logging

import mxnet as mx
import numpy as np
from scipy import stats

import gluonnlp as nlp
import utils


def evaluate_similarity(args, vocab, subword_vocab, embedding_in, subword_net,
                        dataset, similarity_function='CosineSimilarity'):
    """Evaluation on similarity task."""
    initial_length = len(dataset)
    dataset = [
        d for d in dataset
        if d[0] in vocab.token_to_idx and d[1] in vocab.token_to_idx
    ]
    num_dropped = initial_length - len(dataset)
    if num_dropped:
        logging.debug('Dropped %s pairs from %s as the were OOV.', num_dropped,
                      dataset.__class__.__name__)

    dataset_coded = [[
        vocab.token_to_idx[d[0]], vocab.token_to_idx[d[1]], d[2]
    ] for d in dataset]

    if not dataset_coded:
        return 0, 0

    words1, words2, scores = zip(*dataset_coded)

    context = utils.get_context(args)

    # Prepare remapping of indices for use with subwords
    token_bytes, unique_indices = subword_vocab.to_subwords(
        indices=words1 + words2)
    words1 = subword_vocab.remap_indices(unique_indices, words1)
    words2 = subword_vocab.remap_indices(unique_indices, words2)

    # Get vectors from Subword Network
    token_bytes = mx.nd.array(token_bytes, ctx=context[0])
    subword_idx_to_vec = subword_net(token_bytes)

    # Get vectors from TokenEmbedding
    token_idx_to_vec = embedding_in.weight.data(ctx=context[0]).retain(
        mx.nd.array(unique_indices, ctx=context[0])).data

    # Combine vectors
    idx_to_vec = subword_idx_to_vec + token_idx_to_vec

    # Evaluate
    evaluator = nlp.embedding.evaluation.WordEmbeddingSimilarity(
        idx_to_vec=idx_to_vec, similarity_function=similarity_function)
    context = utils.get_context(args)
    evaluator.initialize(ctx=context[0])
    if not args.dont_hybridize:
        evaluator.hybridize()

    pred_similarity = evaluator(
        mx.nd.array(words1, ctx=context[0]), mx.nd.array(
            words2, ctx=context[0]))

    sr = stats.spearmanr(pred_similarity.asnumpy(), np.array(scores))
    logging.debug('Spearman rank correlation on %s: %s',
                  dataset.__class__.__name__, sr.correlation)
    return sr.correlation, len(dataset)


def evaluate_num_zero_rows(args, embedding_in, eps=1E-5):
    context = utils.get_context(args)
    token_idx_to_vec = embedding_in.weight.data(ctx=context[0]).as_in_context(
        mx.cpu()).data
    embedding_norm = mx.nd.norm(token_idx_to_vec, axis=1)
    num_zero_rows = mx.nd.sum(embedding_norm < eps).asscalar()
    return num_zero_rows


def evaluate(args, embedding_in, subword_net, vocab, subword_vocab):
    sr_correlation = 0
    for dataset_name in args.similarity_datasets:
        if subword_net is None:  # TODO Implement
            continue
        if stats is None:
            raise RuntimeError(
                'Similarity evaluation requires scipy.'
                'You may install scipy via `pip install scipy`.')

        logging.debug('Starting evaluation of %s', dataset_name)
        parameters = nlp.data.list_datasets(dataset_name)
        for key_values in itertools.product(*parameters.values()):
            kwargs = dict(zip(parameters.keys(), key_values))
            logging.debug('Evaluating with %s', kwargs)

            dataset = nlp.data.create(dataset_name, **kwargs)
            for similarity_function in args.similarity_functions:
                logging.debug('Evaluating with  %s', similarity_function)
                result, num_samples = evaluate_similarity(
                    args, vocab, subword_vocab, embedding_in, subword_net,
                    dataset, similarity_function)
                sr_correlation += result
    sr_correlation /= len(args.similarity_datasets)

    num_zero_rows = evaluate_num_zero_rows(args, embedding_in)

    return {'SpearmanR': sr_correlation, 'NumZeroRows': num_zero_rows}
