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
"""Argument helpers"""

import argparse
import logging
import os
import sys
import tempfile

import plot
import data
import gluonnlp as nlp
import subword
import trainer
import utils


def get_args(parameter_adders=None):
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        description='Word embedding training with Gluon.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model arguments
    group = parser.add_argument_group('Model arguments')
    group.add_argument('--no-token-embedding', action='store_true',
                       help='Don\'t use any token embedding. '
                       'Only use subword units.')
    group.add_argument(
        '--subword-network', type=str, default='', nargs='?',
        help='Network architecture to encode subword level information. ' +
        str(subword.list_subwordnetworks()))
    group.add_argument('--objective', type=str, default='skipgram',
                       help='Word embedding training objective.')
    group.add_argument('--emsize', type=int, default=300,
                       help='Size of word embeddings')
    group.add_argument(
        '--normalized-initialization', action='store_true',
        help='Normalize uniform initialization range by embedding size.')

    # Evaluation arguments
    group = parser.add_argument_group('Evaluation arguments')
    group.add_argument('--eval-interval', type=int, default=100,
                       help='evaluation interval')
    ## Datasets
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

    # Computation options
    group = parser.add_argument_group('Computation arguments')
    group.add_argument('--batch-size', type=int, default=1024,
                       help='Batch size for training.')
    group.add_argument('--epochs', type=int, default=5, help='Epoch limit')
    group.add_argument('--gpu', type=int, nargs='+',
                       help=('Number (index) of GPU to run on, e.g. 0. '
                             'If not specified, uses CPU.'))
    group.add_argument('--dont-hybridize', action='store_true',
                       help='Disable hybridization of gluon HybridBlocks.')
    group.add_argument('--normalize-gradient', type=str, default='none',
                       help='Normalize the word embedding gradient row-wise. '
                       'Supported are [None, count, L2].')
    group.add_argument(
        '--force-py-op-normalize-gradient', action='store_true',
        help='Always use Python sparse L2 normalization operator.')
    group.add_argument('--use-threaded-data-workers', action='store_true',
                       help='Enable threaded data pre-fetching.')
    group.add_argument('--num-data-workers', type=int, default=5,
                       help='Number of threads to preload data.')


    # Logging options
    group = parser.add_argument_group('Logging arguments')
    group.add_argument('--logdir', type=str, default=None,
                       help='Directory to store logs in.'
                       'Tensorboard compatible logs are stored there. '
                       'Defaults to a random directory in ./logs')

    # Debugging arguments
    group = parser.add_argument_group('Debugging arguments')
    group.add_argument('--debug', action='store_true',
                       help='Enable debug mode checks.')

    # Add further argument groups
    subword.add_parameters(parser)
    data.add_parameters(parser)
    plot.add_parameters(parser)
    trainer.add_parameters(parser)

    if parameter_adders is not None:
        for f in parameter_adders:
            f(parser)

    args = parser.parse_args()

    return args


###############################################################################
# Parse arguments
###############################################################################
def validate_args(args):
    """Validate provided arguments and act on --help."""
    # Check correctness of similarity dataset names

    context = utils.get_context(args)
    assert args.batch_size % len(context) == 0, \
        "Total batch size must be multiple of the number of devices"

    for dataset_name in args.similarity_datasets:
        if dataset_name.lower() not in map(
                str.lower,
                nlp.data.word_embedding_evaluation.word_similarity_datasets):
            print('{} is not a supported dataset.'.format(dataset_name))
            sys.exit(1)

    if args.no_token_embedding and not args.subword_network:
        raise RuntimeError('At least one of token and subword level embedding '
                           'has to be used')


def setup_logging(args):
    """Set up the logging directory."""

    if not args.logdir:
        args.logdir = tempfile.mkdtemp(dir='./logs')
    elif not os.path.isdir(args.logdir):
        os.makedirs(args.logdir)
    elif os.path.isfile(args.logdir):
        raise RuntimeError('{} is a file.'.format(args.logdir))

    logging.info('Logging to {}'.format(args.logdir))


def get_and_setup(*args, **kwargs):
    args_ = get_args(*args, **kwargs)
    validate_args(args_)
    setup_logging(args_)
    return args_
