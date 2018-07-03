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
"""Evaluation
=============

Functions to perform evaluation of TokenEmbeddings on the datasets included in
the GluonNLP toolkit.

"""
import itertools
import sys
import logging

import mxnet as mx
import numpy as np
from scipy import stats

import gluonnlp as nlp


def add_parameters(parser):
    """Add evaluation specific parameters to parser."""
    group = parser.add_argument_group('Evaluation arguments')

    group.add_argument('--eval-batch-size', type=int, default=1024)

    # Datasets
    group.add_argument(
        '--similarity-datasets', type=str,
        default=nlp.data.word_embedding_evaluation.word_similarity_datasets,
        nargs='*',
        help='Word similarity datasets to use for intrinsic evaluation.')
    group.add_argument(
        '--similarity-functions', type=str,
        default=nlp.embedding.evaluation.list_evaluation_functions(
            'similarity'), nargs='+',
        help='Word similarity functions to use for intrinsic evaluation.')
    group.add_argument(
        '--analogy-datasets', type=str, default=['GoogleAnalogyTestSet'],
        nargs='*',
        help='Word similarity datasets to use for intrinsic evaluation.')
    group.add_argument(
        '--analogy-functions', type=str,
        default=nlp.embedding.evaluation.list_evaluation_functions('analogy'),
        nargs='+',
        help='Word analogy functions to use for intrinsic evaluation. ')

    ## Analogy evaluation specific arguments
    group.add_argument(
        '--analogy-dont-exclude-question-words', action='store_true',
        help=('Exclude input words from valid output analogies.'
              'The performance of word embeddings on the analogy task '
              'is around 0% accuracy if input words are not excluded.'))


def validate_args(args):
    """Validate provided arguments and act on --help."""
    # Check correctness of similarity dataset names
    for dataset_name in args.similarity_datasets:
        if dataset_name.lower() not in map(
                str.lower,
                nlp.data.word_embedding_evaluation.word_similarity_datasets):
            print('{} is not a supported dataset.'.format(dataset_name))
            sys.exit(1)

    # Check correctness of analogy dataset names
    for dataset_name in args.analogy_datasets:
        if dataset_name.lower() not in map(
                str.lower,
                nlp.data.word_embedding_evaluation.word_analogy_datasets):
            print('{} is not a supported dataset.'.format(dataset_name))
            sys.exit(1)


def similarity_datasets(dataset_names):
    """Similarity evaluation datasets"""
    datasets = (
        ('WordSim353Similarity', 'similarity',
         nlp.data.create('WordSim353', segment='similarity')),
        ('WordSim353Relatedness', 'relatedness',
         nlp.data.create('WordSim353', segment='relatedness')),
        ('MEN', 'all', nlp.data.create('MEN')),
        ('RadinskyMTurk', 'all', nlp.data.create('RadinskyMTurk')),
        ('RareWords', 'all', nlp.data.create('RareWords')),
        ('SimLex999', 'similarity', nlp.data.create('SimLex999')),
        ('SimVerb3500', 'imilarity', nlp.data.create('SimVerb3500')),
        ('SemEval17Task2', 'similarity', nlp.data.create('SemEval17Task2')),
        ('BakerVerb143', 'all', nlp.data.create('SemEval17Task2')),
        ('YangPowersVerb130', 'all', nlp.data.create('YangPowersVerb130')),
    )

    return [
        d for d in datasets
        if any(d[0].lower().startswith(specified_dataset.lower())
               for specified_dataset in dataset_names)
    ]


def analogy_datasets(dataset_names):
    """Analogy evaluation datasets"""
    datasets = (
        ('GoogleAnalogyTestSet', 'morphological',
         nlp.data.create('GoogleAnalogyTestSet', group='syntactic')),
        ('GoogleAnalogyTestSet', 'semantic',
         nlp.data.create('GoogleAnalogyTestSet', group='semantic')),
    )

    return [
        d for d in datasets
        if any(d[0].lower().startswith(specified_dataset.lower())
               for specified_dataset in dataset_names)
    ]


def similarity_dataset_split_categories(datasets, vocab,
                                        zero_word_vectors_set):
    """Split similarity datasets into parts.

    Based on if a sample contains only known words, known words but of which
    some are in zero_word_vectors_set or contains any unknown (ie OOV) words.

    Returns
    -------
    list of tuple(str, str, str, nlp.data.WordSimilarityEvaluationDataset)
      name, word_type, category, dataset where name, word_type are copied from
      input and category is in {'all', 'all_known', 'known_but_zero',
      'unknown'}

    """

    category_datasets = []
    for name, word_type, dataset in datasets:
        all_known = []
        known_but_zero = []
        unknown = []
        for sample in dataset:
            if any(w not in vocab for w in sample[:2]):
                unknown.append(sample)
            elif any(w in zero_word_vectors_set for w in sample[:2]):
                known_but_zero.append(sample)
            else:
                all_known.append(sample)
        category_datasets.append((name, word_type, 'all', dataset))
        if all_known:
            category_datasets.append((name, word_type, 'all_known', all_known))
        if known_but_zero:
            category_datasets.append((name, word_type, 'known_but_zero',
                                      known_but_zero))
        if unknown:
            category_datasets.append((name, word_type, 'unknown', unknown))
    return category_datasets


def analogy_dataset_split_categories(datasets, vocab, zero_word_vectors_set):
    """Split similarity datasets into parts.

    Based on if a sample contains only known words, known words but of which
    some are in zero_word_vectors_set or contains any unknown (ie OOV) words.

    Returns
    -------
    list of tuple(str, str, str, nlp.data.WordSimilarityEvaluationDataset)
      name, word_type, category, dataset where name, word_type are copied from
      input and category is in {'all_known', 'known_but_zero', 'unknown'}

    """
    category_datasets = []
    for name, word_type, dataset in datasets:
        all_known = []
        known_but_zero = []
        unknown = []
        for sample in dataset:
            if any(w not in vocab for w in sample):
                unknown.append(sample)
            elif any(w in zero_word_vectors_set for w in sample):
                known_but_zero.append(sample)
            else:
                all_known.append(sample)
        if all_known:
            category_datasets.append((name, word_type, 'all_known', all_known))
        if known_but_zero:
            category_datasets.append((name, word_type, 'known_but_zero',
                                      known_but_zero))
        if unknown:
            category_datasets.append((name, word_type, 'unknown', unknown))
    return category_datasets


def get_tokens_in_evaluation_datasets(args):
    """Returns a set of all tokens occuring the evaluation datasets."""
    tokens = set()
    for _, _, dataset in similarity_datasets(args.similarity_datasets):
        tokens.update(
            itertools.chain.from_iterable((d[0], d[1]) for d in dataset))

    for _, _, dataset in analogy_datasets(args.analogy_datasets):
        tokens.update(
            itertools.chain.from_iterable(
                (d[0], d[1], d[2], d[3]) for d in dataset))

    return tokens


def evaluate_similarity(args, token_embedding, ctx, vocab,
                        zero_word_vectors_set={}, logfile=None, global_step=0):
    """Evaluate on specified similarity datasets."""

    datasets = similarity_dataset_split_categories(
        similarity_datasets(args.similarity_datasets), vocab,
        zero_word_vectors_set)

    for similarity_function in args.similarity_functions:
        evaluator = nlp.embedding.evaluation.WordEmbeddingSimilarity(
            idx_to_vec=token_embedding.idx_to_vec,
            similarity_function=similarity_function)
        evaluator.initialize(ctx=ctx)
        if not args.no_hybridize:
            evaluator.hybridize()

        # Evaluate all datasets
        for (dataset_name, word_type, category, dataset) in datasets:
            initial_length = len(dataset)
            dataset_coded = [[
                token_embedding.token_to_idx[d[0]],
                token_embedding.token_to_idx[d[1]], d[2]
            ] for d in dataset if d[0] in token_embedding.token_to_idx
                             and d[1] in token_embedding.token_to_idx]
            num_dropped = initial_length - len(dataset_coded)

            words1, words2, scores = zip(*dataset_coded)
            pred_similarity = evaluator(
                mx.nd.array(words1, ctx=ctx), mx.nd.array(words2, ctx=ctx))
            sr = stats.spearmanr(pred_similarity.asnumpy(), np.array(scores))
            logging.info('Spearman rank correlation on %s %s %s with %s:\t%s',
                         dataset_name, word_type, category,
                         similarity_function, sr.correlation)

            log_result(logfile, 'similarity', dataset_name, word_type,
                       category, similarity_function, sr.correlation,
                       len(dataset_coded), num_dropped, global_step)


def evaluate_analogy(args, token_embedding, ctx, vocab,
                     zero_word_vectors_set={}, logfile=None, global_step=0):
    """Evaluate on specified analogy datasets.

    The analogy task is an open vocabulary task, make sure to pass a
    token_embedding with a sufficiently large number of supported tokens.

    """
    datasets = analogy_dataset_split_categories(
        analogy_datasets(args.analogy_datasets), vocab, zero_word_vectors_set)
    exclude_question_words = not args.analogy_dont_exclude_question_words
    for analogy_function in args.analogy_functions:
        evaluator = nlp.embedding.evaluation.WordEmbeddingAnalogy(
            idx_to_vec=token_embedding.idx_to_vec,
            exclude_question_words=exclude_question_words,
            analogy_function=analogy_function)
        evaluator.initialize(ctx=ctx)
        if not args.no_hybridize:
            evaluator.hybridize()

        for (dataset_name, word_type, category, dataset) in datasets:
            initial_length = len(dataset)
            dataset_coded = [[
                token_embedding.token_to_idx[d[0]],
                token_embedding.token_to_idx[d[1]],
                token_embedding.token_to_idx[d[2]],
                token_embedding.token_to_idx[d[3]]
            ] for d in dataset if d[0] in token_embedding.token_to_idx
                             and d[1] in token_embedding.token_to_idx
                             and d[2] in token_embedding.token_to_idx
                             and d[3] in token_embedding.token_to_idx]
            num_dropped = initial_length - len(dataset_coded)

            dataset_coded_batched = mx.gluon.data.DataLoader(
                dataset_coded, batch_size=args.eval_batch_size)

            acc = mx.metric.Accuracy()
            for batch in dataset_coded_batched:
                batch = batch.as_in_context(ctx)
                words1, words2, words3, words4 = (batch[:, 0], batch[:, 1],
                                                  batch[:, 2], batch[:, 3])
                pred_idxs = evaluator(words1, words2, words3)
                acc.update(pred_idxs[:, 0], words4.astype(np.float32))

            logging.info('Accuracy on %s %s %s with %s:\t%s', dataset_name,
                         word_type, category, analogy_function,
                         acc.get()[1])

            log_result(logfile, 'similarity', dataset_name, word_type,
                       category, analogy_function,
                       acc.get()[1], len(dataset_coded), num_dropped,
                       global_step)


def log_result(logfile, *args):
    """Log to a file.

    Parameters
    ----------
    logfile : str
      Path of file to append results to. If not logfile this function does
      nothing.
    *args : tuple(str)
      Values to log
    """
    if not logfile:
        return

    with open(logfile, 'a') as f:
        f.write('\t'.join(str(v) for v in args))
        f.write('\n')
