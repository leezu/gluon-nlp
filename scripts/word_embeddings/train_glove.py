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

# pylint: disable=global-variable-undefined,wrong-import-position
"""GloVe embedding model
===========================

This example shows how to train a GloVe embedding model based on the vocabulary
and co-occurrence matrix constructed by the vocab_count and cooccur tool. The
tools are located in the same ./tools folder next to this script.

The GloVe model was introduced by

- Pennington, J., Socher, R., & Manning, C. D. (2014). Glove: global vectors
  for word representation. In A. Moschitti, B. Pang, & W. Daelemans,
  Proceedings of the 2014 Conference on Empirical Methods in Natural Language
  Processing, {EMNLP} 2014, October 25-29, 2014, Doha, Qatar, {A} meeting of
  SIGDAT, a Special Interest Group of the {ACL (pp. 1532–1543). : ACL.

"""
# * Imports
import argparse
import io
import logging
import os
import random
import sys
import tempfile
import time

import mxnet as mx
import numpy as np

import evaluation
import gluonnlp as nlp
from gluonnlp.base import _str_types
from utils import get_context, print_time

os.environ['MXNET_GPU_MEM_POOL_TYPE'] = 'Round'


# * Utils
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='GloVe with GluonNLP',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Data options
    group = parser.add_argument_group('Data arguments')
    group.add_argument(
        'cooccurrences', type=str,
        help='Path to cooccurrences.npz containing a sparse (COO) '
        'representation of the co-occurrence matrix in numpy archive format. '
        'Output of ./cooccur')
    group.add_argument('vocab', type=str,
                       help='Vocabulary indices. Output of vocab_count tool.')

    # Computation options
    group = parser.add_argument_group('Computation arguments')
    group.add_argument('--batch-size', type=int, default=65536,
                       help='Batch size for training.')
    group.add_argument('--epochs', type=int, default=50, help='Epoch limit')
    group.add_argument(
        '--gpu', type=int, nargs='+',
        help='Number (index) of GPU to run on, e.g. 0. '
        'If not specified, uses CPU.')
    group.add_argument('--no-hybridize', action='store_true',
                       help='Disable hybridization of gluon HybridBlocks.')
    group.add_argument(
        '--no-static-alloc', action='store_true',
        help='Disable static memory allocation for HybridBlocks.')

    # Model
    group = parser.add_argument_group('Model arguments')
    group.add_argument('--emsize', type=int, default=300,
                       help='Size of embedding vectors.')
    group.add_argument('--x-max', type=int, default=100)
    group.add_argument('--alpha', type=float, default=0.75)

    # Optimization options
    group = parser.add_argument_group('Optimization arguments')
    group.add_argument('--adagrad-eps', type=float, default=1,
                       help='Initial AdaGrad state value.')
    group.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    group.add_argument('--seed', type=int, default=1, help='Random seed')
    group.add_argument('--dropout', type=float, default=0.15)

    # Logging
    group = parser.add_argument_group('Logging arguments')
    group.add_argument('--logdir', type=str, default='logs',
                       help='Directory to store logs.')
    group.add_argument('--log-interval', type=int, default=100)
    group.add_argument(
        '--eval-interval', type=int,
        help='Evaluate every --eval-interval iterations '
        'in addition to at the end of every epoch.')
    group.add_argument('--no-eval-analogy', action='store_true',
                       help='Don\'t evaluate on the analogy task.')

    # Evaluation options
    evaluation.add_parameters(parser)

    args = parser.parse_args()
    evaluation.validate_args(args)

    random.seed(args.seed)
    mx.random.seed(args.seed)
    np.random.seed(args.seed)

    return args


def get_train_data(args):
    """Helper function to get training data."""
    counter = dict()
    with io.open(args.vocab, 'r', encoding='utf-8') as f:
        for line in f:
            token, count = line.split('\t')
            counter[token] = int(count)
    vocab = nlp.Vocab(counter, unknown_token=None, padding_token=None,
                      bos_token=None, eos_token=None, min_freq=1)

    npz = np.load(args.cooccurrences)
    row, col, counts = npz['row'], npz['col'], npz['data']

    rank_dtype = 'int32'
    if row.max() >= np.iinfo(np.int32).max:
        rank_dtype = 'int64'
        # MXNet has no support for uint32, so we must fall back to int64
        logging.info('More words than could be counted using int32. '
                     'Using int64 to represent word indices.')
    row = mx.nd.array(row, dtype=rank_dtype)
    col = mx.nd.array(col, dtype=rank_dtype)
    # row is always used as 'source' and col as 'context' word. Therefore
    # duplicate the entries.

    assert row.shape == col.shape
    row = mx.nd.concatenate([row, col])
    col = mx.nd.concatenate([col, row[:len(row) // 2]])

    counts = mx.nd.array(counts, dtype='float32')
    counts = mx.nd.concatenate([counts, counts])

    return vocab, row, col, counts


# * Gluon Block definition
class GloVe(nlp.model.train.EmbeddingModel, mx.gluon.HybridBlock):
    """GloVe EmbeddingModel"""

    def __init__(self, token_to_idx, output_dim, x_max, alpha, dropout=0,
                 weight_initializer=None,
                 bias_initializer=mx.initializer.Zero(), sparse_grad=True,
                 dtype='float32', **kwargs):
        assert isinstance(token_to_idx, dict)

        super(GloVe, self).__init__(**kwargs)
        self.token_to_idx = token_to_idx
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.sparse_grad = sparse_grad
        self.dtype = dtype

        self._x_max = x_max
        self._alpha = alpha
        self._dropout = dropout

        with self.name_scope():
            self.source_embedding = mx.gluon.nn.Embedding(
                len(token_to_idx), output_dim,
                weight_initializer=weight_initializer, sparse_grad=sparse_grad,
                dtype=dtype)
            self.context_embedding = mx.gluon.nn.Embedding(
                len(token_to_idx), output_dim,
                weight_initializer=weight_initializer, sparse_grad=sparse_grad,
                dtype=dtype)
            self.source_bias = mx.gluon.nn.Embedding(
                len(token_to_idx), 1, weight_initializer=bias_initializer,
                sparse_grad=sparse_grad, dtype=dtype)
            self.context_bias = mx.gluon.nn.Embedding(
                len(token_to_idx), 1, weight_initializer=bias_initializer,
                sparse_grad=sparse_grad, dtype=dtype)

    def hybrid_forward(self, F, row, col, counts):
        """Compute embedding of words in batch.

        Parameters
        ----------
        row : mxnet.nd.NDArray or mxnet.sym.Symbol
            Array of token indices for source words. Shape (batch_size, ).
        row : mxnet.nd.NDArray or mxnet.sym.Symbol
            Array of token indices for context words. Shape (batch_size, ).
        counts : mxnet.nd.NDArray or mxnet.sym.Symbol
            Their co-occurrence counts. Shape (batch_size, ).

        Returns
        -------
        mxnet.nd.NDArray or mxnet.sym.Symbol
            Loss. Shape (batch_size, ).

        """

        emb_in = self.source_embedding(row)
        emb_out = self.context_embedding(col)

        if self._dropout:
            emb_in = F.Dropout(emb_in, p=self._dropout)
            emb_out = F.Dropout(emb_out, p=self._dropout)

        bias_in = self.source_bias(row).squeeze()
        bias_out = self.context_bias(col).squeeze()
        dot = F.batch_dot(emb_in.expand_dims(1),
                          emb_out.expand_dims(2)).squeeze()
        tmp = dot + bias_in + bias_out - F.log(counts).squeeze()
        weight = F.clip(((counts / self._x_max)**self._alpha), a_min=0,
                        a_max=1).squeeze()
        loss = weight * F.square(tmp)
        return loss

    def __contains__(self, token):
        return token in self.idx_to_token

    def __getitem__(self, tokens):
        """Looks up embedding vectors of text tokens.

        Parameters
        ----------
        tokens : str or list of strs
            A token or a list of tokens.

        Returns
        -------
        mxnet.ndarray.NDArray:
            The embedding vector(s) of the token(s). According to numpy
            conventions, if `tokens` is a string, returns a 1-D NDArray
            (vector); if `tokens` is a list of strings, returns a 2-D NDArray
            (matrix) of shape=(len(tokens), vec_len).
        """
        squeeze = False
        if isinstance(tokens, _str_types):
            tokens = [tokens]
            squeeze = True

        indices = mx.nd.array([self.token_to_idx[t] for t in tokens],
                              ctx=self.source_embedding.weight.list_ctx()[0])
        vecs = self.source_embedding(indices) + self.context_embedding(indices)

        if squeeze:
            assert len(vecs) == 1
            return vecs[0].squeeze()
        else:
            return vecs


# * Training code
def train(args):
    """Training helper."""
    vocab, row, col, counts = get_train_data(args)
    model = GloVe(token_to_idx=vocab.token_to_idx, output_dim=args.emsize,
                  dropout=args.dropout, x_max=args.x_max, alpha=args.alpha,
                  weight_initializer=mx.init.Uniform(scale=1 / args.emsize))
    context = get_context(args)
    model.initialize(ctx=context)
    if not args.no_hybridize:
        model.hybridize(static_alloc=not args.no_static_alloc)

    optimizer_kwargs = dict(learning_rate=args.lr, eps=args.adagrad_eps)
    params = list(model.collect_params().values())
    try:
        trainer = mx.gluon.Trainer(params, 'groupadagrad', optimizer_kwargs)
    except ValueError:
        logging.warning('MXNet <= v1.3 does not contain '
                        'GroupAdaGrad support. Falling back to AdaGrad')
        trainer = mx.gluon.Trainer(params, 'adagrad', optimizer_kwargs)

    index_dtype = 'int32'
    if counts.shape[0] >= np.iinfo(np.int32).max:
        index_dtype = 'int64'
        logging.info('Co-occurrence matrix is large. '
                     'Using int64 to represent sample indices.')
    indices = mx.nd.arange(counts.shape[0], dtype=index_dtype)
    for epoch in range(args.epochs):
        # Logging variables
        log_wc = 0
        log_start_time = time.time()
        log_avg_loss = 0

        mx.nd.shuffle(indices, indices)  # inplace shuffle
        bs = args.batch_size
        num_batches = indices.shape[0] // bs
        for i in range(num_batches):
            batch_indices = indices[bs * i:bs * (i + 1)]
            ctx = context[i % len(context)]
            batch_row = row[batch_indices].as_in_context(ctx)
            batch_col = col[batch_indices].as_in_context(ctx)
            batch_counts = counts[batch_indices].as_in_context(ctx)
            with mx.autograd.record():
                loss = model(batch_row, batch_col, batch_counts)
                loss.backward()

            if len(context) == 1 or (i + 1) % len(context) == 0:
                trainer.step(batch_size=1)

            # Logging
            log_wc += loss.shape[0]
            log_avg_loss += loss.mean().as_in_context(context[0])
            if (i + 1) % args.log_interval == 0:
                # Forces waiting for computation by computing loss value
                log_avg_loss = log_avg_loss.asscalar() / args.log_interval
                wps = log_wc / (time.time() - log_start_time)
                logging.info('[Epoch {} Batch {}/{}] loss={:.4f}, '
                             'throughput={:.2f}K wps, wc={:.2f}K'.format(
                                 epoch, i + 1, num_batches, log_avg_loss,
                                 wps / 1000, log_wc / 1000))
                log_dict = dict(
                    global_step=epoch * len(indices) + i * args.batch_size,
                    epoch=epoch, batch=i + 1, loss=log_avg_loss,
                    wps=wps / 1000)
                log(args, log_dict)

                log_start_time = time.time()
                log_avg_loss = 0
                log_wc = 0

            if args.eval_interval and (i + 1) % args.eval_interval == 0:
                with print_time('mx.nd.waitall()'):
                    mx.nd.waitall()
                with print_time('evaluate'):
                    evaluate(args, model, vocab, i + num_batches * epoch)

    # Evaluate
    with print_time('mx.nd.waitall()'):
        mx.nd.waitall()
    with print_time('evaluate'):
        evaluate(args, model, vocab, num_batches * args.epochs,
                 eval_analogy=not args.no_eval_analogy)

    # Save params
    with print_time('save parameters'):
        model.save_parameters(os.path.join(args.logdir, 'glove.params'))


# * Evaluation
def evaluate(args, model, indexer, global_step, eval_analogy=False):
    """Evaluation helper"""
    if 'eval_tokens' not in globals():
        global eval_tokens

        eval_tokens_set = evaluation.get_tokens_in_evaluation_datasets(args)
        if not args.no_eval_analogy:
            eval_tokens_set.update(indexer.index_to_word)

        # GloVe does not support computing vectors for OOV words
        eval_tokens_set = filter(lambda t: t in indexer.word_to_index,
                                 eval_tokens_set)

        eval_tokens = list(eval_tokens_set)

    # Compute their word vectors
    context = get_context(args)
    mx.nd.waitall()

    token_embedding = nlp.embedding.TokenEmbedding(unknown_token=None,
                                                   allow_extend=True)
    token_embedding[eval_tokens] = model[eval_tokens]

    results = evaluation.evaluate_similarity(
        args, token_embedding, context[0], logfile=os.path.join(
            args.logdir, 'similarity.tsv'), global_step=global_step)
    if eval_analogy:
        assert not args.no_eval_analogy
        results += evaluation.evaluate_analogy(
            args, token_embedding, context[0], logfile=os.path.join(
                args.logdir, 'analogy.tsv'))

    return results


# * Logging
def log(args, kwargs):
    """Log to a file."""
    logfile = os.path.join(args.logdir, 'log.tsv')

    if 'log_created' not in globals():
        if os.path.exists(logfile):
            logging.error('Logfile %s already exists.', logfile)
            sys.exit(1)

        global log_created

        log_created = sorted(kwargs.keys())
        header = '\t'.join((str(k) for k in log_created)) + '\n'
        with open(logfile, 'w') as f:
            f.write(header)

    # Log variables shouldn't change during training
    assert log_created == sorted(kwargs.keys())

    with open(logfile, 'a') as f:
        f.write('\t'.join((str(kwargs[k]) for k in log_created)) + '\n')


# * External
import logging
import re
from collections import Counter

import numpy as np
import torch
from sklearn.datasets import fetch_20newsgroups
from torch.autograd import Variable
from torch.utils.data import Dataset
from tqdm import tqdm

# Hyperparameters
N_EMBEDDING = 300
BASE_STD = 0.01
BATCH_SIZE = 512
NUM_EPOCH = 10
MIN_WORD_OCCURENCES = 10
X_MAX = 100
ALPHA = 0.75
BETA = 0.0001
RIGHT_WINDOW = 15

USE_CUDA = True
# USE_CUDA = False

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT)


def cuda(x):
    if USE_CUDA:
        return x.cuda()
    return x


class WordIndexer:
    """Transform g a dataset of text to a list of index of words. Not memory 
    optimized for big datasets"""

    def __init__(self, min_word_occurences=1, right_window=1, oov_word="OOV"):
        self.oov_word = oov_word
        self.right_window = right_window
        self.min_word_occurences = min_word_occurences
        self.word_to_index = {oov_word: 0}
        self.index_to_word = [oov_word]
        self.word_occurrences = {}
        self.re_words = re.compile(r"\b[a-zA-Z]{2,}\b")

    def _get_or_set_word_to_index(self, word):
        try:
            return self.word_to_index[word]
        except KeyError:
            idx = len(self.word_to_index)
            self.word_to_index[word] = idx
            self.index_to_word.append(word)
            return idx

    @property
    def n_words(self):
        return len(self.word_to_index)

    def fit_transform(self, texts):
        l_words = [
            list(self.re_words.findall(sentence.lower()))
            for sentence in texts]
        word_occurrences = Counter(word for words in l_words for word in words)

        self.word_occurrences = {
            word: n_occurences
            for word, n_occurences in word_occurrences.items()
            if n_occurences >= self.min_word_occurences}

        oov_index = 0
        return [[
            self._get_or_set_word_to_index(word)
            if word in self.word_occurrences else oov_index for word in words]
                for words in l_words]

    def _get_ngrams(self, indexes):
        for i, left_index in enumerate(indexes):
            window = indexes[i + 1:i + self.right_window + 1]
            for distance, right_index in enumerate(window):
                yield left_index, right_index, distance + 1

    def get_comatrix(self, data):
        comatrix = Counter()
        z = 0
        for indexes in data:
            l_ngrams = self._get_ngrams(indexes)
            for left_index, right_index, distance in l_ngrams:
                comatrix[(left_index, right_index)] += 1. / distance
                z += 1
        return zip(*[(left, right, x)
                     for (left, right), x in comatrix.items()])


class GloveDataset(Dataset):
    def __len__(self):
        return self.n_obs

    def __getitem__(self, index):
        raise NotImplementedError()

    def __init__(self, texts, right_window=1, random_state=0):
        torch.manual_seed(random_state)

        self.indexer = WordIndexer(right_window=right_window,
                                   min_word_occurences=MIN_WORD_OCCURENCES)
        data = self.indexer.fit_transform(texts)
        left, right, n_occurrences = self.indexer.get_comatrix(data)
        n_occurrences = np.array(n_occurrences)
        self.n_obs = len(left)

        # We create the variables
        self.L_words = cuda(torch.LongTensor(left))
        self.R_words = cuda(torch.LongTensor(right))

        self.weights = np.minimum((n_occurrences / X_MAX)**ALPHA, 1)
        self.weights = Variable(cuda(torch.FloatTensor(self.weights)))
        self.y = Variable(cuda(torch.FloatTensor(np.log(n_occurrences))))

        # We create the embeddings and biases
        N_WORDS = self.indexer.n_words
        L_vecs = cuda(torch.randn((N_WORDS, N_EMBEDDING)) * BASE_STD)
        R_vecs = cuda(torch.randn((N_WORDS, N_EMBEDDING)) * BASE_STD)
        L_biases = cuda(torch.randn((N_WORDS, )) * BASE_STD)
        R_biases = cuda(torch.randn((N_WORDS, )) * BASE_STD)
        self.all_params = [
            Variable(e, requires_grad=True)
            for e in (L_vecs, R_vecs, L_biases, R_biases)]
        self.L_vecs, self.R_vecs, self.L_biases, self.R_biases = self.all_params


def gen_batchs(data):
    """Batch sampling function"""
    indices = torch.randperm(len(data))
    if USE_CUDA:
        indices = indices.cuda()
    for idx in range(0, len(data) - BATCH_SIZE + 1, BATCH_SIZE):
        sample = indices[idx:idx + BATCH_SIZE]
        l_words, r_words = data.L_words[sample], data.R_words[sample]
        l_vecs = data.L_vecs[l_words]
        r_vecs = data.R_vecs[r_words]
        l_bias = data.L_biases[l_words]
        r_bias = data.R_biases[r_words]
        weight = data.weights[sample]
        y = data.y[sample]
        yield weight, l_vecs, r_vecs, y, l_bias, r_bias


def get_loss(weight, l_vecs, r_vecs, log_covals, l_bias, r_bias):
    sim = (l_vecs * r_vecs).sum(1).view(-1)
    x = (sim + l_bias + r_bias - log_covals)**2
    loss = torch.mul(x, weight)
    return loss.mean()


def train_model(data: GloveDataset):
    optimizer = torch.optim.Adam(data.all_params, weight_decay=1e-8)
    optimizer.zero_grad()
    for epoch in tqdm(range(NUM_EPOCH)):
        logging.info("Start epoch %i", epoch)
        num_batches = int(len(data) / BATCH_SIZE)
        avg_loss = 0.0
        n_batch = int(len(data) / BATCH_SIZE)
        for batch in tqdm(gen_batchs(data), total=n_batch, mininterval=1):
            optimizer.zero_grad()
            loss = get_loss(*batch)
            avg_loss += loss.data[0] / num_batches
            loss.backward()
            optimizer.step()
        logging.info("Average loss for epoch %i: %.5f", epoch + 1, avg_loss)


def train_model_gluon(data: GloveDataset, model):
    params = model.collect_params()
    trainer = mx.gluon.Trainer(params, 'adam', dict(wd=1e-8))
    for epoch in tqdm(range(NUM_EPOCH)):
        logging.info("Start epoch %i", epoch)
        num_batches = int(len(data) / BATCH_SIZE)
        avg_loss = 0.0
        n_batch = int(len(data) / BATCH_SIZE)
        for batch in tqdm(gen_batchs(data), total=n_batch, mininterval=1):
            batch = [
                mx.nd.array(np.asarray(batch[i].detach()))
                for i in range(len(batch))]
            import ipdb
            ipdb.set_trace()
            with mx.autograd.record():
                loss = model(batch_row, batch_col, batch_counts)
                loss.backward()
            avg_loss += loss.data[0] / num_batches
            loss.backward()
            trainer.step()
        logging.info("Average loss for epoch %i: %.5f", epoch + 1, avg_loss)


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    args_ = parse_args()

    logging.info("Fetching data")
    # newsgroup = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))
    # logging.info("Build dataset")
    data = nlp.data.Text8()
    data = [' '.join(s) for s in data]
    glove_data = GloveDataset(data, right_window=RIGHT_WINDOW)
    logging.info("#Words: %s", glove_data.indexer.n_words)
    logging.info("#Ngrams: %s", len(glove_data))
    logging.info("Start training")

    model = GloVe(token_to_idx=glove_data.indexer.word_to_index,
                  output_dim=args_.emsize, dropout=args_.dropout,
                  x_max=args_.x_max, alpha=args_.alpha,
                  weight_initializer=mx.init.Uniform(scale=1 / args_.emsize))
    context = get_context(args_)
    model.initialize(ctx=context)

    train_model(glove_data)
    # train_model_gluon(glove_data, model)

    model.source_embedding.weight.set_data(
        mx.nd.array(np.asarray(glove_data.all_params[0].detach())))
    model.context_embedding.weight.set_data(
        mx.nd.array(np.asarray(glove_data.all_params[1].detach())))

    if os.path.exists(args_.logdir):
        newlogdir = tempfile.mkdtemp(dir=args_.logdir)
        logging.warning('%s exists. Using %s', args_.logdir, newlogdir)
        args_.logdir = newlogdir
    if not os.path.isdir(args_.logdir):
        os.makedirs(args_.logdir)
    evaluate(args_, model, glove_data.indexer, 0, eval_analogy=True)
