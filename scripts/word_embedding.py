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

import argparse
import time

import mxnet as mx
import tqdm
from mxnet import gluon

import gluonnlp as nlp

parser = argparse.ArgumentParser(
    description='Word embedding training with Gluon.')

possible_embedding_names = [
    "{k}:{v}".format(k=k, v=v) for k in nlp.embedding.list_sources().keys()
    for v in nlp.embedding.list_sources()[k]
]
parser.add_argument(
    '--embedding-name',
    type=str,
    default='glove:glove.6B.300d.txt',
    help=('Name of embedding type to load. '
          'Valid entries: {}'.format(", ".join(possible_embedding_names))))

# Evaluation options
parser.add_argument(
    '--eval-similarity',
    type=str,
    default='*',
    nargs='+',
    help='Word similarity datasets to use for intrinsic evaluation. '
    'Defaults to all (wildcard "*")')
parser.add_argument(
    '--disable-eval-nearest-neighbors',
    action='store_false',
    help='Print nearest neighbors of 5 random words in SimVerb3500')

# Hyperparameters
parser.add_argument(
    '--normalized_initialization',
    action='store_true',
    help='Normalize uniform initialization range by embedding size.')

# Temporary arguments
parser.add_argument('--old-batching-behaviour', action='store_true')
parser.add_argument('--eval-size', type=int, default=5)

args = parser.parse_args()

print(args)

###############################################################################
# Build the model
###############################################################################

embedding_name, pretrained_file_name = args.embedding_name.split(":")

print("Load embedding ", args.embedding_name)

token_embedding = text.embedding.create(
    embedding_name, pretrained_file_name=pretrained_file_name)

###############################################################################
# Evaluation code
###############################################################################
evaluators = []

# Word similarity based evaluation
if args.eval_similarity:
    similarity_datasets = \
        nlp.data.word_embedding_evaluation.word_similarity_datasets
    if args.eval_similarity == "*":
        args.eval_similarity = similarity_datasets

    for ds in args.eval_similarity:
        if ds not in similarity_datasets:
            print(("{ds} is not a supported dataset. "
                   "Only {supported} are supported").format(
                       ds=ds, supported=", ".join(similarity_datasets)))
            continue

        ds_class = eval("nlp.data.{ds}".format(ds=ds))
        evaluator = nlp.evaluation.WordEmbeddingSimilarityEvaluator(
            dataset=ds_class(), token_to_idx=sgdataset._token_to_idx)
        if len(evaluator):
            evaluators.append(evaluator)

# Nearest neighbor printing based evaluation
if not args.disable_eval_nearest_neighbors:
    nn_evaluator = nlp.evaluation.WordEmbeddingNearestNeighborEvaluator(
        dataset=nlp.data.SimVerb3500(), token_to_idx=sgdataset._token_to_idx)


def evaluate():
    eval_dict = {}
    for evaluator in evaluators:
        score = evaluator(net.embedding_in)
        eval_dict[evaluator.dataset.__class__.__name__] = score

    return eval_dict


###############################################################################
# Training code
###############################################################################


def train():
    for epoch in range(args.epochs):
        if not args.disable_eval_nearest_neighbors:
            nn_evaluator(net.embedding_in)

        import time
        if args.old_batching_behaviour:
            start = time.time()
            data_loader = mx.gluon.data.DataLoader(
                sgdataset,
                batch_size=args.batch_size,
                shuffle=True,
                last_batch='discard',
                num_workers=args.num_data_loader_workers)
            print(f"Spent {time.time() - start} building data_loader")
            t = tqdm.trange(len(data_loader))
        else:
            start = time.time()
            sampler = gluon.data.RandomSampler(len(sgdataset))
            print(f"Spent {time.time() - start} building sampler")
            start = time.time()
            batch_sampler = gluon.data.BatchSampler(sampler, args.batch_size,
                                                    'discard')
            print(f"Spent {time.time() - start} building batch_sampler")
            data_loader = (sgdataset[batch] for batch in batch_sampler)
            t = tqdm.trange(len(batch_sampler))

        for i, (source, target, label) in zip(t, data_loader):
            source = mx.nd.array(source, ctx=context[0])
            target = mx.nd.array(target, ctx=context[0])
            label = mx.nd.array(label, ctx=context[0])

            with mx.autograd.record():
                pred = net(source, target)
                l = loss(pred, label)
            l.backward()
            trainer.step(batch_size=1)

            if i % args.eval_interval == 0:
                eval_dict = evaluate()

            t.set_postfix(
                # TODO print number of grad norm > 0
                loss=mx.nd.sum(l).asscalar(),
                grad=net.embedding_in.weight.grad().norm().asscalar(),
                data=net.embedding_in.weight.data().norm(
                    axis=1).mean().asscalar(),
                **eval_dict)


if __name__ == '__main__':
    start_pipeline_time = time.time()
    if not args.eval_only:
        train()
