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
import functools
import itertools
import logging
import math
import multiprocessing as mp
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import attr
import mxnet as mx
import numpy as np
import tqdm
from mxnet import gluon
from mxnet.gluon import nn, Block
from scipy import stats

import gluonnlp as nlp
import sparse_ops

try:
    import ujson as json
except ImportError:
    logging.warning('ujson not installed. '
                    ' Install via `pip install ujson` '
                    'for faster data preprocessing.')

try:
    import tqdm
except ImportError:
    logging.warning('tqdm not installed. '
                    ' Install via `pip install tqdm` for better usability.')
    tqdm = None


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
    group.add_argument('--lr', type=float, default=0.1,
                       help='Initial learning rate')
    group.add_argument('--epochs', type=int, default=5, help='Epoch limit')
    group.add_argument('--gpu', type=int, nargs='+',
                       help=('Number (index) of GPU to run on, e.g. 0. '
                             'If not specified, uses CPU.'))
    group.add_argument('--dont-hybridize', action='store_true',
                       help='Disable hybridization of gluon HybridBlocks.')
    group.add_argument('--dont-normalize-gradient', action='store_true',
                       help='L2 normalize word embedding gradients per word.')
    group.add_argument(
        '--force-py-op-normalize-gradient', action='store_true',
        help='Always use Python sparse L2 normalization operator.')

    # Logging options
    group = parser.add_argument_group('Logging arguments')
    group.add_argument(
        '--log', type=str, default='results.csv', help='Path to logfile.'
        'Results of evaluation runs are written to there in a CSV format.')

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
class SubwordRNN(Block):
    """RNN model for sub-word embedding inference.

    Parameters
    ----------
    mode : str
        The type of RNN to use. Options are 'lstm', 'gru', 'rnn_tanh',
        'rnn_relu'.
    embed_size : int
        Dimension of embedding vectors for subword units.
    hidden_size : int
        Number of hidden units for RNN.
    output_size : int
        Dimension of embedding vectors for subword units.
    num_layers : int
        Number of RNN layers.
    vocab_size : int, default 2**8
        Size of the input vocabulary. Usually the input vocabulary is the
        number of distinct bytes.
    dropout : float
        Dropout rate to use for encoder output.

    """

    def __init__(self, mode, embed_size, hidden_size, num_layers, output_size,
                 vocab_size=256, dropout=0.5, **kwargs):
        super(SubwordRNN, self).__init__(**kwargs)
        self._mode = mode
        self._embed_size = embed_size
        self._hidden_size = hidden_size
        self._output_size = output_size
        self._num_layers = num_layers
        self._dropout = dropout
        self._vocab_size = vocab_size

        with self.name_scope():
            self.embedding = self._get_embedding()
            self.encoder = self._get_encoder()
            self.decoder = self._get_decoder()

    def _get_embedding(self):
        embedding = nn.HybridSequential()
        with embedding.name_scope():
            embedding.add(
                nn.Embedding(self._vocab_size, self._embed_size,
                             weight_initializer=mx.init.Uniform(0.1)))
            if self._dropout:
                embedding.add(nn.Dropout(self._dropout))
        return embedding

    def _get_encoder(self):
        return nlp.model.utils._get_rnn_layer(
            self._mode, self._num_layers, self._embed_size, self._hidden_size,
            self._dropout, 0)

    def _get_decoder(self):
        output = nn.HybridSequential()
        with output.name_scope():
            output.add(nn.Dense(self._output_size, flatten=False))
        return output

    def begin_state(self, *args, **kwargs):
        return self.encoder.begin_state(*args, **kwargs)

    def forward(self, inputs, begin_state=None):  # pylint: disable=arguments-differ
        """Defines the forward computation. Arguments can be either
        :py:class:`NDArray` or :py:class:`Symbol`."""
        encoded = self.embedding(inputs)
        if not begin_state:
            begin_state = self.begin_state(batch_size=inputs.shape[1],
                                           ctx=inputs.context)
        encoded, state = self.encoder(encoded, begin_state)
        if self._dropout:
            encoded = mx.nd.Dropout(encoded, p=self._dropout, axes=(0, ))
        out = self.decoder(encoded)
        return out, state


class SkipGramModel(gluon.Block):
    def __init__(self, num_tokens, embedding_dim, **kwargs):
        super().__init__(**kwargs)

        self.num_tokens = num_tokens
        self.embedding_dim = embedding_dim

        with self.name_scope():
            # Training of embedding lookup matrix
            self.embedding_in = gluon.nn.SparseEmbedding(
                self.num_tokens, self.embedding_dim)
            self.embedding_out = gluon.nn.SparseEmbedding(
                self.num_tokens, self.embedding_dim)

            # Training of subword embedding function
            self.subword = SubwordRNN(mode='lstm', embed_size=32,
                                      hidden_size=64,
                                      output_size=embedding_dim, num_layers=1)

    def forward(self, source, target, token_bytes, source_subword,
                target_subword, F=mx.nd):
        word_emb_in = self.embedding_in(source)
        word_emb_out = self.embedding_out(target)

        subword_emb_weights = mx.nd.max(self.subword(token_bytes)[0], axis=1)

        subword_emb_in = mx.nd.Embedding(
            data=source_subword, weight=subword_emb_weights,
            input_dim=subword_emb_weights.shape[0],
            output_dim=self.embedding_dim)
        subword_emb_out = mx.nd.Embedding(
            data=target_subword, weight=subword_emb_weights,
            input_dim=subword_emb_weights.shape[0],
            output_dim=self.embedding_dim)

        emb_in = word_emb_in + subword_emb_in
        emb_out = word_emb_out + subword_emb_out

        return F.batch_dot(emb_in, emb_out.swapaxes(1, 2))


###############################################################################
# Load data
###############################################################################
@contextmanager
def print_time(task):
    start_time = time.time()
    logging.info('Starting to {}'.format(task))
    yield
    logging.info('Finished to {} in {} seconds'.format(
        task,
        time.time() - start_time))


def token_to_index(serialized_sentences, vocab):
    sentences = json.loads(serialized_sentences)
    coded = [
        np.array([vocab[token] for token in s
                  if token in vocab], dtype=np.int32) for s in sentences
    ]
    return coded


def get_train_data(args):
    # TODO currently only supports skipgram and a single dataset
    # → Add Text8 Dataset to the toolkit
    with print_time('read dataset to memory'):
        sentences = nlp.data.Text8(segment='train')

    # Count tokens
    with print_time('count all tokens'):
        counter = nlp.data.count_tokens(
            itertools.chain.from_iterable(sentences))

    vocab = nlp.Vocab(counter, unknown_token=None, padding_token=None,
                      bos_token=None, eos_token=None)

    # Split dataset into parts for fast multiprocessing (and serialize with
    # json to avoid slow pickle operation)
    num_workers = mp.cpu_count()
    if len(sentences) == 1:
        size = math.ceil(len(sentences[0]) / num_workers)
        worker_sentences = [[sentences[0][i:i + size]]
                            for i in range(0, len(sentences[0]), size)]
    else:
        size = math.ceil(len(sentences) / num_workers)
        worker_sentences = [[sentences[i:i + size]]
                            for i in range(0, len(sentences), size)]

    worker_sentences = [json.dumps(s) for s in worker_sentences]
    with mp.Pool(processes=num_workers) as pool:
        with print_time('code all sentences'):
            coded = pool.map(
                functools.partial(token_to_index, vocab=vocab),
                worker_sentences)
            coded = sum(coded, [])
            if len(sentences) == 1:
                coded = [np.concatenate(coded)]

    # Prune frequent words from sentences
    with print_time('prune frequent words from sentences'):
        frequent_tokens_subsampling_constant = 1e-3
        idx_to_counts = np.array(vocab.idx_to_counts, dtype=int)
        f = idx_to_counts / np.sum(idx_to_counts)
        idx_to_pdiscard = (np.sqrt(frequent_tokens_subsampling_constant / f) +
                           frequent_tokens_subsampling_constant / f)

        # prune_sentences releases GIL so multi-threading is sufficient
        prune_sentences = functools.partial(
            nlp.data.word_embedding_training.prune_sentences,
            idx_to_pdiscard=idx_to_pdiscard)
        with ThreadPoolExecutor(max_workers=num_workers) as e:
            coded = list(e.map(prune_sentences, coded))

    # Get index to byte mapping from vocab
    idx_to_bytes = [
        np.frombuffer(token.encode('utf-8'), dtype=np.uint8)
        for token in vocab.idx_to_token
    ]

    sgdataset = nlp.data.SkipGramWordEmbeddingDataset(coded, idx_to_counts,
                                                      idx_to_bytes)
    return sgdataset, vocab


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

    return net, loss


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


def evaluate(args, net, vocab):
    context = get_context(args)
    token_to_idx = vocab.token_to_idx
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
    train_dataset, vocab = get_train_data(args)
    net, loss = get_model(args, train_dataset)
    context = get_context(args)

    params = list(net.collect_params().values())
    sparse_params = [p for p in params if 'sparseembedding' in p.name]
    dense_params = [p for p in params if 'sparseembedding' not in p.name]

    if len(context) > 1:
        sparse_kvstore = mx.kv.create('device')
    else:
        sparse_kvstore = None

    dense_trainer = gluon.Trainer(dense_params, 'sgd',
                                  {'learning_rate': args.lr})

    _kv_initialized = False
    indices = np.arange(len(train_dataset))
    for epoch in range(args.epochs):
        np.random.shuffle(indices)
        batches = [
            indices[i:i + args.batch_size]
            for i in range(0, len(indices), args.batch_size)
        ]

        if tqdm is not None:
            t = tqdm.trange(len(batches), smoothing=1)
        else:
            t = range(len(batches))

        num_workers = math.ceil(mp.cpu_count() * 0.8)
        executor = ThreadPoolExecutor(max_workers=num_workers)
        for i, (source, target, label,
                token_bytes, source_subword, target_subword) in zip(
                    t, executor.map(train_dataset.__getitem__, batches)):
            source = gluon.utils.split_and_load(source, context)
            target = gluon.utils.split_and_load(target, context)
            label = gluon.utils.split_and_load(label, context)

            assert len(context) == 1  # TODO now only 1 GPU supported
            token_bytes = mx.nd.array(token_bytes, ctx=context[0])
            source_subword = mx.nd.array(source_subword, ctx=context[0])
            target_subword = mx.nd.array(target_subword, ctx=context[0])

            with mx.autograd.record():
                losses = [
                    loss(
                        net(X, Y, token_bytes, source_subword, target_subword),
                        Z) for X, Y, Z in zip(source, target, label)
                ]
            for l in losses:
                l.backward()

            if not _kv_initialized and len(context) > 1:
                for param_i, param in enumerate(params):
                    sparse_kvstore.init(param_i, param.list_data()[0])
                _kv_initialized = True

            # Training of dense params
            dense_trainer.step(batch_size=args.batch_size)

            # Training of sparse params
            for param_i, param in enumerate(sparse_params):
                if param.grad_req == 'null':
                    continue

                # Synchronize devices
                if len(context) > 1:
                    sparse_kvstore.push(param_i, param.list_grad(),
                                        priority=-param_i)

                    # Get indices updated rows
                    row_ids = mx.nd.concat(*[
                        p.indices.as_in_context(mx.cpu())
                        for p in param.list_grad()
                    ], dim=0)

                    # Share gradients
                    sparse_kvstore.row_sparse_pull(param_i, param.list_grad(),
                                                   priority=-param_i,
                                                   row_ids=row_ids)

                # Update of sparse params
                for device_param, device_grad in zip(param.list_data(),
                                                     param.list_grad()):
                    if args.dont_normalize_gradient:
                        pass
                    elif (hasattr(mx.nd.sparse, 'l2_normalization')
                          and not args.force_py_op_normalize_gradient):
                        norm = mx.nd.sparse.sqrt(
                            mx.nd._internal._square_sum(
                                device_grad, axis=1, keepdims=True))
                        mx.nd.sparse.l2_normalization(device_grad, norm,
                                                      out=device_grad)
                    else:
                        device_grad = mx.nd.Custom(
                            device_grad, op_type='sparse_l2normalization')

                    mx.nd.sparse.sgd_update(weight=device_param,
                                            grad=device_grad, lr=args.lr,
                                            out=device_param)

            if i % args.eval_interval == 0:
                eval_dict = evaluate(args, net, vocab)

                t.set_postfix(
                    # TODO print number of grad norm > 0
                    loss=sum(l.sum().as_in_context(mx.cpu())
                             for l in losses).asscalar(),
                    grad=sum(
                        net.embedding_in.weight.grad(
                            ctx=ctx).as_in_context(mx.cpu()).norm()
                        for ctx in context).asscalar(),
                    dense_grad=sum(
                        p.grad(ctx=context[0]).norm()
                        for p in dense_params).asscalar(),
                    data=net.embedding_in.weight.data(
                        ctx=context[0]).as_in_context(
                            mx.cpu()).tostype("default").norm(
                                axis=1).mean().asscalar(),
                    **eval_dict)

        executor.shutdown()


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    if not hasattr(mx.nd.sparse, 'l2_normalization'):
        logging.warning('Mxnet version is not compiled with '
                        'sparse l2_normalization support. '
                        ' Using slow Python implementation.')

    args_ = get_args()
    validate_args(args_)

    train(args_)
