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


def construct_vocab_embedding_for_dataset(args, tokens, vocab, embedding_in,
                                          subword_vocab=None,
                                          subword_net=None):
    '''Precompute the token embeddings for all the words in the vocabulary'''
    assert len(tokens) == len(set(tokens)), 'tokens contains duplicates.'

    known_tokens = []
    context = utils.get_context(args)
    token_subword_embeddings = []
    if subword_vocab is not None:
        words_subword_indices = subword_vocab.words_to_subwordindices(tokens)
        for token, subword_indices in zip(tokens, words_subword_indices):
            subword_indices_nd = mx.nd.array(subword_indices, ctx=context[0])

            # If no subwords are associated with this token
            if not sum(subword_indices_nd.shape):
                # If token is also not in vocabulary
                if token not in vocab:
                    continue
                else:
                    known_tokens.append(token)
                    token_subword_embeddings.append(
                        mx.nd.zeros((1, embedding_in.weight.shape[1]),
                                    ctx=context[0]))
                    continue

            if subword_net is not None:
                # Add batch dimension and infer token_subword_embedding
                subword_indices_nd = mx.nd.expand_dims(subword_indices_nd, 0)
                mask = mx.nd.ones_like(subword_indices_nd)
                if subword_indices_nd.shape[1] < subword_net.subword.min_size:
                    missing = (subword_net.subword.min_size -
                               subword_indices_nd.shape[1])
                    subword_indices_nd = mx.nd.concat(
                        subword_indices_nd,
                        mx.nd.zeros((1, missing),
                                    ctx=subword_indices_nd.context))
                    mask = mx.nd.concat(mask,
                                        mx.nd.zeros((1, missing),
                                                    ctx=mask.context))
                token_subword_embedding = subword_net(subword_indices_nd, mask)
                token_subword_embeddings.append(token_subword_embedding)
            else:  # Subword indices should be applicable for embedding_in
                subword_embeddings = embedding_in(subword_indices_nd)
                token_subword_embedding = mx.nd.sum(subword_embeddings, axis=0)
                token_subword_embedding = mx.nd.expand_dims(
                    token_subword_embedding, 0)
                token_subword_embeddings.append(token_subword_embedding)

            known_tokens.append(token)
    else:
        assert subword_net is None, \
            'If subword_net is supplied, ' \
            'also a subword_vocab needs to be specified.'

    # Get token embeddings from embedding_in
    token_to_idx = {token: i for i, token in enumerate(known_tokens)}
    token_embeddings = []
    for token in known_tokens:
        if token in vocab:
            token_embedding = embedding_in(
                mx.nd.array([vocab.to_indices(token)], ctx=context[0]))
            token_embeddings.append(token_embedding)
        else:
            token_embeddings.append(
                mx.nd.zeros((1, embedding_in.weight.shape[1]), ctx=context[0]))

    # Combine subword and word level embeddings
    if token_subword_embeddings:
        assert token_embeddings[0].shape == \
            token_subword_embeddings[0].shape
        embeddings = mx.nd.concat(*token_embeddings, dim=0) + \
            mx.nd.concat(*token_subword_embeddings, dim=0)
    elif token_embeddings:
        assert len(token_embeddings[0].shape) == 2
        embeddings = mx.nd.concat(*token_embeddings, dim=0)
    else:
        # Didn't get any embeddings
        embeddings = mx.nd.array([], ctx=context[0])

    return token_to_idx, embeddings


def _filter_similarity_dataset(token_to_idx, dataset):
    initial_length = len(dataset)
    dataset = [
        d for d in dataset if d[0] in token_to_idx and d[1] in token_to_idx
    ]

    num_dropped = initial_length - len(dataset)
    if num_dropped:
        logging.debug('Dropped %s pairs from %s as the were OOV.', num_dropped,
                      dataset.__class__.__name__)
    return dataset


def evaluate_similarity(args, vocab, subword_vocab, embedding_in, subword_net,
                        dataset, similarity_function='CosineSimilarity'):
    """Evaluation on similarity task."""
    tokens = set(itertools.chain.from_iterable((d[0], d[1]) for d in dataset))
    tokens = list(tokens)

    # Get token embedding matrix based on word and subword information
    token_to_idx, idx_to_vec = construct_vocab_embedding_for_dataset(
        args, tokens, vocab, embedding_in, subword_vocab, subword_net)
    dataset = _filter_similarity_dataset(token_to_idx, dataset)
    dataset_coded = [[token_to_idx[d[0]], token_to_idx[d[1]], d[2]]
                     for d in dataset]

    if not dataset_coded:
        logging.info('Dataset {} contains only OOV. Skipping.'.format(
            dataset.__class__.__name__))
        return 0, 0

    # Evaluate
    words1, words2, scores = zip(*dataset_coded)
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
