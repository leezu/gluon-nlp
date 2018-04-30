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
import random
import time

import attr
import mxnet as mx
import tqdm
from mxnet import gluon

import gluonnlp as nlp

parser = argparse.ArgumentParser(
    description='Word embedding training with Gluon.')
parser.add_argument(
    '--emsize', type=int, default=300, help='size of word embeddings')
parser.add_argument(
    '--lr', type=float, default=0.1, help='initial learning rate')
parser.add_argument('--epochs', type=int, default=5, help='upper epoch limit')
parser.add_argument(
    '--batch-size', type=int, default=1024, metavar='N', help='batch size')
parser.add_argument(
    '--num-data-loader-workers',
    type=int,
    default=0,
    help='Number of workers preparing batches.')
parser.add_argument(
    '--log-interval',
    type=int,
    default=200,
    metavar='N',
    help='report interval')
parser.add_argument(
    '--eval-interval', type=int, default=1000, help='evaluation interval')
parser.add_argument(
    '--save',
    type=str,
    default='model.params',
    help='path to save the final model')
parser.add_argument(
    '--eval-only',
    action='store_true',
    help='Whether to only evaluate the trained model')
parser.add_argument(
    '--gpus',
    type=str,
    help=
    'list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu. (the result of multi-gpu training might be slightly different compared to single-gpu training, still need to be finalized)'
)

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
        super().__init__(
            prefix=self.gluon_prefix, params=self.gluon_params, **kwargs)

        with self.name_scope():
            self.embedding_in = gluon.nn.Embedding(self.num_tokens,
                                                   self.embedding_dim)
            self.embedding_out = gluon.nn.Embedding(self.num_tokens,
                                                    self.embedding_dim)

    def hybrid_forward(self, F, source, target):
        emb_in = self.embedding_in(source)
        emb_out = self.embedding_out(target)
        return F.batch_dot(emb_in, emb_out.swapaxes(1, 2))


###############################################################################
# Load data
###############################################################################

context = [mx.cpu()] if args.gpus is None or args.gpus == "" else \
          [mx.gpu(int(i)) for i in args.gpus.split(',')]

assert len(context) == 1

assert args.batch_size % len(
    context) == 0, "Total batch size must be multiple of the number of devices"

text8 = nlp.data.Text8(segment='train')
sgdataset = nlp.data.SkipGramWordEmbeddingDataset(text8)
train_dataset = sgdataset

###############################################################################
# Build the model
###############################################################################

num_tokens = sgdataset.num_tokens

net = SkipGramModel(num_tokens=num_tokens, embedding_dim=args.emsize)
if args.normalized_initialization:
    net.initialize(mx.init.Uniform(scale=1 / args.emsize), ctx=context[0])
else:
    net.initialize(mx.init.Uniform(), ctx=context[0])
loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
net.hybridize()

trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'learning_rate': args.lr})

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
