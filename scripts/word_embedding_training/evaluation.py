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
    assert embedding_in is not None or subword_net is not None

    context = utils.get_context(args)
    token_subword_embeddings = []

    known_tokens = []
    known_tokens_subwordindices = []
    token_mask = []

    if subword_vocab is not None:
        words_subword_indices = subword_vocab.words_to_subwordindices(tokens)

        # Collect tokens for which either word or subword embeddings are known
        for token, subword_indices in zip(tokens, words_subword_indices):
            # If no subwords are associated with this token
            if (not len(subword_indices) and vocab is not None
                    and token not in vocab):
                continue

            known_tokens.append(token)
            # subword_indices may be empty list
            known_tokens_subwordindices.append(subword_indices)

            if vocab is None:
                continue
            elif token in vocab:
                token_mask.append(1)
            else:
                token_mask.append(0)

        # Compute embeddings based on subword units
        # 1. Create batch and mask for padding
        max_num_subwords = max(len(s) for s in known_tokens_subwordindices)
        if subword_net is not None:
            assert max_num_subwords > subword_net.min_size, \
                'All words have less subwords then the required minimum length. '\
                'Looks like a bug.'  # Check git blame to find padding code
        known_tokens_subwordindices_np = np.zeros(
            (len(known_tokens_subwordindices), max_num_subwords))
        known_tokens_subwordindices_mask_np = np.zeros(
            (len(known_tokens_subwordindices), max_num_subwords))
        for i, subword_indices in enumerate(known_tokens_subwordindices):
            if not len(subword_indices):
                continue
            known_tokens_subwordindices_np[i, :len(subword_indices)] = \
                subword_indices
            known_tokens_subwordindices_mask_np[i, :len(subword_indices)] = 1

        # 2. Copy to device
        known_tokens_subword_indices_nd = mx.nd.array(
            known_tokens_subwordindices_np, ctx=context[0])
        known_tokens_subword_indices_mask_nd = mx.nd.array(
            known_tokens_subwordindices_mask_np, ctx=context[0])
        known_tokens_subword_indices_last_valid = \
                (known_tokens_subwordindices_mask_np == 0).argmax(axis=1) - 1
        known_tokens_subword_indices_last_valid[
            known_tokens_subword_indices_last_valid ==
            -1] = known_tokens_subwordindices_mask_np.shape[1] - 1
        known_tokens_subword_indices_last_valid_nd = mx.nd.array(
            known_tokens_subword_indices_last_valid, ctx=context[0])

        # 3. Compute
        if subword_net is not None:
            token_subword_embeddings, _ = subword_net(
                known_tokens_subword_indices_nd,
                known_tokens_subword_indices_mask_nd,
                known_tokens_subword_indices_last_valid_nd)
        else:  # Subword indices should be applicable for embedding_in
            subword_embeddings = embedding_in(known_tokens_subword_indices_nd)
            masked_subword_embeddings = mx.nd.broadcast_mul(
                subword_embeddings,
                known_tokens_subword_indices_mask_nd.expand_dims(-1))
            token_subword_embeddings = mx.nd.sum(masked_subword_embeddings,
                                                 axis=-2)

    else:
        assert subword_net is None, \
            'If subword_net is supplied, ' \
            'also a subword_vocab needs to be specified.'
        for token in tokens:
            if token in vocab:
                known_tokens.append(token)
                token_mask.append(1)
            else:
                # As we don't have a subword_vocab, we can't handle tokens that
                # are not in the vocab and skip them
                continue

    # Compute result token_to_idx map
    token_to_idx = {token: i for i, token in enumerate(known_tokens)}

    # Look up token embeddings from embedding_in
    if embedding_in is not None:
        known_token_idx = [
            vocab.to_indices(t) if t in vocab else 0 for t in known_tokens
        ]
        known_token_idx_nd = mx.nd.array(known_token_idx, ctx=context[0])
        known_token_idx_mask_nd = mx.nd.array(token_mask, ctx=context[0]) \
                                       .expand_dims(-1)
        unmasked_token_embeddings = embedding_in(known_token_idx_nd)
        token_embeddings = mx.nd.broadcast_mul(unmasked_token_embeddings,
                                               known_token_idx_mask_nd)

    # Combine subword and word level embeddings
    if subword_vocab is not None and embedding_in is not None:
        assert token_embeddings.shape == \
            token_subword_embeddings.shape
        embeddings = token_embeddings + token_subword_embeddings
    elif subword_vocab is not None:
        embeddings = token_subword_embeddings
    else:
        embeddings = token_embeddings

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
                        dataset_name, dataset_kwargs,
                        similarity_function='CosineSimilarity',
                        mxboard_summary_writer=None):
    """Evaluation on similarity task."""
    dataset = nlp.data.create(dataset_name, **dataset_kwargs)
    dataset_name_wkwargs = dataset_name + ','.join(
        "{!s}={!r}".format(k, v) for (k, v) in dataset_kwargs.items())

    tokens = set(itertools.chain.from_iterable((d[0], d[1]) for d in dataset))
    tokens = list(tokens)

    # Get token embedding matrix based on word and subword information
    token_to_idx, idx_to_vec = construct_vocab_embedding_for_dataset(
        args, tokens, vocab, embedding_in, subword_vocab, subword_net)
    dataset = _filter_similarity_dataset(token_to_idx, dataset)
    dataset_coded = [[token_to_idx[d[0]], token_to_idx[d[1]], d[2]]
                     for d in dataset]

    if mxboard_summary_writer is not None:
        idx_to_token = [
            x[0] for x in sorted(token_to_idx.items(), key=lambda x: x[1])
        ]
        mxboard_summary_writer.add_embedding(
            tag='similarity-{}'.format(dataset_name_wkwargs),
            embedding=idx_to_vec, labels=idx_to_token)

    if not dataset_coded:
        logging.info('Dataset {} contains only OOV. Skipping.'.format(
            dataset_name_wkwargs))
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
    return num_zero_rows, embedding_norm.shape[0]


def evaluate(args, embedding_in, subword_net, vocab, subword_vocab,
             mxboard_summary_writer=None):
    eval_dict = {}
    for dataset_name in args.similarity_datasets:
        if stats is None:
            raise RuntimeError(
                'Similarity evaluation requires scipy.'
                'You may install scipy via `pip install scipy`.')

        logging.debug('Starting evaluation of %s', dataset_name)
        parameters = nlp.data.list_datasets(dataset_name)
        for key_values in itertools.product(*parameters.values()):
            dataset_kwargs = dict(zip(parameters.keys(), key_values))
            logging.debug('Evaluating with %s', dataset_kwargs)

            for similarity_function in args.similarity_functions:
                logging.debug('Evaluating with  %s', similarity_function)
                result, num_words = evaluate_similarity(
                    args, vocab, subword_vocab, embedding_in, subword_net,
                    dataset_name, dataset_kwargs, similarity_function,
                    mxboard_summary_writer)
                dataset_name_wkwargs = dataset_name + ','.join(
                    "{!s}={!r}".format(k, v)
                    for (k, v) in dataset_kwargs.items())
                eval_dict['similarity-sr-' + dataset_name_wkwargs] = result
                eval_dict['similarity-numwords--'
                          + dataset_name_wkwargs] = num_words

    if embedding_in is not None:
        num_zero_rows, num_total_rows = evaluate_num_zero_rows(
            args, embedding_in)
        return {
            'Zero': num_zero_rows / num_total_rows,
            'Total': num_total_rows,
            **eval_dict
        }

    else:
        return eval_dict
