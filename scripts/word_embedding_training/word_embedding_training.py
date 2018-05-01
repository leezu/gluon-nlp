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

import argparse
import logging
import sys
import itertools

import attr
import mxnet as mx
import tqdm
from mxnet import gluon
import numpy as np
from scipy import stats

import gluonnlp as nlp

try:
    import progressbar
except ImportError:
    logging.warning(
        'progressbar not installed. '
        ' Install via `pip install progressbar2` for better usability.')
    progressbar = None


def get_args():
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        description='Word embedding training with Gluon.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model arguments
    group = parser.add_argument_group('Model arguments')
    group.add_argument('--emsize', type=int, default=300,
                       help='Size of word embeddings')
    group.add_argument(
        '--normalized-initialization', action='store_true',
        help='Normalize uniform initialization range by embedding size.')
    # Evaluation arguments
    group = parser.add_argument_group('Evaluation arguments')
    group.add_argument('--eval-interval', type=int, default=1000,
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
    group.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training.')
    group.add_argument('--lr', type=float, default=0.1,
                       help='Initial learning rate')
    group.add_argument('--epochs', type=int, default=5, help='Epoch limit')
    group.add_argument('--gpu', type=int, nargs='+',
                       help=('Number (index) of GPU to run on, e.g. 0. '
                             'If not specified, uses CPU.'))
    group.add_argument('--dont-hybridize', action='store_true',
                       help='Disable hybridization of gluon HybridBlocks.')

    # Logging options
    group = parser.add_argument_group('Logging arguments')
    group.add_argument(
        '--log', type=str, default='results.csv', help='Path to logfile.'
        'Results of evaluation runs are written to there in a CSV format.')

    # Temporary arguments  TODO remove
    group = parser.add_argument_group('Temporary arguments')
    group.add_argument('--old-batching-behaviour', action='store_true')
    group.add_argument('--eval-size', type=int, default=5)

    args = parser.parse_args()

    return args


###############################################################################
# Parse arguments
###############################################################################
def validate_args(args):
    """Validate provided arguments and act on --help."""
    # Check correctness of similarity dataset names

    context = get_context(args)
    assert args.batch_size % len(context) == 0, \
        "Total batch size must be multiple of the number of devices"

    for dataset_name in args.similarity_datasets:
        if dataset_name.lower() not in map(
                str.lower,
                nlp.data.word_embedding_evaluation.word_similarity_datasets):
            print('{} is not a supported dataset.'.format(dataset_name))
            sys.exit(1)


def get_context(args):
    if args.gpu is None or args.gpu == '':
        context = [mx.cpu()]
    else:
        context = [mx.gpu(int(i)) for i in args.gpu]
    return context


###############################################################################
# Model definitions
###############################################################################
@attr.s
class SkipGramModel(gluon.HybridBlock):
    num_tokens = attr.ib()
    embedding_dim = attr.ib()

    # gluon.Block parameters
    gluon_prefix = attr.ib(default=None)
    gluon_params = attr.ib(default=None)

    def __attrs_post_init__(self, **kwargs):
        super().__init__(prefix=self.gluon_prefix, params=self.gluon_params,
                         **kwargs)

        with self.name_scope():
            self.embedding_in = gluon.nn.SparseEmbedding(
                self.num_tokens, self.embedding_dim)
            self.embedding_out = gluon.nn.SparseEmbedding(
                self.num_tokens, self.embedding_dim)

    def hybrid_forward(self, F, source, target):
        emb_in = self.embedding_in(source)
        emb_out = self.embedding_out(target)
        return F.batch_dot(emb_in, emb_out.swapaxes(1, 2))


###############################################################################
# Load data
###############################################################################
def get_train_data(args):
    # TODO currently only supports skipgram and a single dataset
    # → Add Text8 Dataset to the toolkit
    text8 = nlp.data.Text8(segment='train')
    sgdataset = nlp.data.SkipGramWordEmbeddingDataset(text8)
    return sgdataset


###############################################################################
# Build the model
###############################################################################
def get_model(args, train_dataset):
    num_tokens = train_dataset.num_tokens

    net = SkipGramModel(num_tokens=num_tokens, embedding_dim=args.emsize)

    context = get_context(args)
    if args.normalized_initialization:
        net.initialize(mx.init.Uniform(scale=1 / args.emsize), ctx=context)
    else:
        net.initialize(mx.init.Uniform(), ctx=context)
    loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
    net.hybridize()

    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': args.lr})

    return net, loss, trainer


###############################################################################
# Evaluation code
###############################################################################
def evaluate_similarity(args, idx_to_vec, token_to_idx, dataset,
                        similarity_function='CosineSimilarity'):
    """Evaluation on similarity task."""
    initial_length = len(dataset)
    dataset = [
        d for d in dataset if d[0] in token_to_idx and d[1] in token_to_idx
    ]
    num_dropped = initial_length - len(dataset)
    if num_dropped:
        logging.debug('Dropped %s pairs from %s as the were OOV.', num_dropped,
                      dataset.__class__.__name__)

    dataset_coded = [[token_to_idx[d[0]], token_to_idx[d[1]], d[2]]
                     for d in dataset]
    words1, words2, scores = zip(*dataset_coded)

    evaluator = nlp.embedding.evaluation.WordEmbeddingSimilarity(
        idx_to_vec=idx_to_vec, similarity_function=similarity_function)
    context = get_context(args)
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


def evaluate(args, net, training_dataset):
    context = get_context(args)
    token_to_idx = training_dataset._token_to_idx
    idx_to_vec = net.embedding_in.weight.data(ctx=context[0])

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
                    args, idx_to_vec, token_to_idx, dataset,
                    similarity_function)
                sr_correlation += result
    sr_correlation /= len(args.similarity_datasets)
    return {'SpearmanR': sr_correlation}


###############################################################################
# Training code
###############################################################################
def train(args):
    train_dataset = get_train_data(args)
    net, loss, trainer = get_model(args, train_dataset)
    context = get_context(args)

    for epoch in range(args.epochs):

        import time
        if args.old_batching_behaviour:
            start = time.time()
            data_loader = mx.gluon.data.DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True,
                last_batch='discard', num_workers=args.num_data_loader_workers)
            print(f"Spent {time.time() - start} building data_loader")
            t = tqdm.trange(len(data_loader))
        else:
            start = time.time()
            sampler = gluon.data.RandomSampler(len(train_dataset))
            print(f"Spent {time.time() - start} building sampler")
            start = time.time()
            batch_sampler = gluon.data.BatchSampler(sampler, args.batch_size,
                                                    'discard')
            print(f"Spent {time.time() - start} building batch_sampler")
            data_loader = (train_dataset[batch] for batch in batch_sampler)
            t = tqdm.trange(len(batch_sampler))

        for i, (source, target, label) in zip(t, data_loader):
            source = gluon.utils.split_and_load(source, context)
            target = gluon.utils.split_and_load(target, context)
            label = gluon.utils.split_and_load(label, context)

            with mx.autograd.record():
                losses = [
                    loss(net(X, Y), Z)
                    for X, Y, Z in zip(source, target, label)
                ]
            for l in losses:
                l.backward()
            trainer.step(batch_size=1)

            if i % args.eval_interval == 0:
                eval_dict = evaluate(args, net, train_dataset)

                t.set_postfix(
                    # TODO print number of grad norm > 0
                    loss=sum(l.sum().as_in_context(mx.cpu())
                             for l in losses).asscalar(),
                    grad=sum(
                        net.embedding_in.weight.grad(
                            ctx=ctx).as_in_context(mx.cpu()).norm()
                        for ctx in context).asscalar(),
                    data=net.embedding_in.weight.data(
                        ctx=context[0]).as_in_context(
                            mx.cpu()).tostype("default").norm(
                                axis=1).mean().asscalar(),
                    **eval_dict)


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    args_ = get_args()
    validate_args(args_)

    train(args_)
