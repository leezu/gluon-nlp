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
"""Subword embedding
====================

"""
import functools
import warnings

import mxnet as mx
from mxnet import gluon

import gluonnlp as nlp
import self_attention
from blocks import VDCNN


###############################################################################
# Hyperparameters
###############################################################################
def add_parameters(parser):
    group = parser.add_argument_group('Subword networks')
    group.add_argument('--subword-embedding-size', type=int, default=16,
                       help='Embedding size for each subword piece.')
    group.add_argument('--subword-embedding-dropout', type=float, default=0.0,
                       help='Embedding size for each subword piece.')

    _selfattention_args(parser)
    _last_output_args(parser)
    _subwordrnn_args(parser)
    _subwordcnn_args(parser)
    _subwordvdcnn_args(parser)
    _awdrnn_args(parser)
    _wordprediction_args(parser)


def _selfattention_args(parser):
    group = parser.add_argument_group('Subword Embedding: Self Attention')
    group.add_argument('--self-attention-num-units', type=int, default=350)
    group.add_argument('--self-attention-num-attention', type=int, default=10)
    group.add_argument('--self-attention-dropout', type=float, default=0.0)
    group.add_argument('--self-attention-regularizer-lambda', type=float,
                       default=1.0)


def _last_output_args(parser):
    group = parser.add_argument_group('Subword Embedding: Last Output')
    group.add_argument('--last-output-hidden-size', type=int, default=1000)


def _subwordrnn_args(parser):
    group = parser.add_argument_group('Subword Network: SubwordRNN')
    group.add_argument('--subwordrnn-mode', type=str, default='lstm')
    group.add_argument('--subwordrnn-hidden-size', type=int, default=1000)
    group.add_argument('--subwordrnn-num-layers', type=int, default=3)
    group.add_argument('--subwordrnn-encoder-dropout', type=float, default=0.0)
    group.add_argument('--subwordrnn-no-bidirectional', default=False,
                       action='store_true')


def _awdrnn_args(parser):
    group = parser.add_argument_group('Subword Network: AWDRNN')
    group.add_argument('--awdrnn-path', type=str, default='model.params',
                       help='Path to pretrained parameters.')
    group.add_argument('--awdrnn-model', type=str, default='lstm',
                       help='type of recurrent net '
                       '(rnn_tanh, rnn_relu, lstm, gru)')
    group.add_argument('--awdrnn-nhid', type=int, default=1150,
                       help='number of hidden units per layer')
    group.add_argument('--awdrnn-nlayers', type=int, default=3,
                       help='number of layers')


def _subwordcnn_args(parser):
    group = parser.add_argument_group('Subword Network: SubwordCNN')
    group.add_argument('--subwordcnn-filter-sizes', type=int, nargs='+',
                       default=[3, 4, 5])
    group.add_argument('--subwordcnn-cnn-dropout', type=float, default=0.0)


def _subwordvdcnn_args(parser):
    group = parser.add_argument_group('Subword Network: SubwordVDCNN')
    group.add_argument('--subwordvdcnn-depth', type=int, default=9)
    group.add_argument('--subwordvdcnn-no-batch-norm', action='store_true')


def _wordprediction_args(parser):
    group = parser.add_argument_group('Auxilary word prediction task.')
    group.add_argument('--wordprediction-hidden-size', type=int, default=300)
    group.add_argument('--wordprediction-activation', type=str, default='tanh')


###############################################################################
# Registry
###############################################################################
_REGSITRY_NAME = {}


def register(class_=None):
    """Registers a model."""

    def _real_register(class_):
        # Save the kwargs associated with this class_
        _REGSITRY_NAME[class_.__name__] = class_

        register_ = mx.registry.get_register_func(SubwordNetwork,
                                                  'subwordnetwork')
        return register_(class_)

    if class_ is not None:
        # Decorator was called without arguments
        return _real_register(class_)

    return _real_register


def create(name, args, **kwargs):
    """Creates an instance of a registered network."""
    create_ = mx.registry.get_create_func(SubwordNetwork, 'subwordnetwork')

    # Network specific arguments
    if name.lower() == 'subwordrnn':
        kwargs = dict(
            embed_size=args.subword_embedding_size,
            embedding_dropout=args.subword_embedding_dropout,
            output_size=args.emsize,
            mode=args.subwordrnn_mode,
            hidden_size=args.subwordrnn_hidden_size,
            num_layers=args.subwordrnn_num_layers,
            bidirectional=not args.subwordrnn_no_bidirectional,
            encoder_dropout=args.subwordrnn_encoder_dropout,
            **kwargs,
        )
    elif name.lower() == 'subwordcnn':
        kwargs = dict(
            embed_size=args.subword_embedding_size,
            embedding_dropout=args.subword_embedding_dropout,
            output_size=args.emsize,
            filter_sizes=args.subwordcnn_filter_sizes,
            cnn_dropout=args.subwordcnn_cnn_dropout,
            **kwargs,
        )
    elif name.lower() == 'subwordvdcnn':
        kwargs = dict(
            embed_size=args.subword_embedding_size,
            embedding_dropout=args.subword_embedding_dropout,
            output_size=args.emsize,
            depth=args.subwordvdcnn_depth,
            temporal_batchnorm=not args.subwordvdcnn_no_batch_norm,
            **kwargs,
        )
    elif name.lower() == 'awdrnn':
        kwargs = dict(
            embed_size=args.subword_embedding_size,
            output_size=args.emsize,
            model=args.awdrnn_model,
            hidden_size=args.awdrnn_nhid,
            num_layers=args.awdrnn_nlayers,
            awdrnn_path=args.awdrnn_path,
            **kwargs,
        )
    elif name.lower() in ['sumreduce', 'meanreduce', 'fasttext']:
        if args.subword_embedding_size != args.emsize:
            warnings.warn('In {} mode, subword-embedding-size '
                          'must equal emsize. Using emsize {} instead.')
        kwargs = dict(embed_size=args.emsize,
                      sparse_embedding=not args.no_use_sparse_embedding,
                      **kwargs)
    elif name.lower() == 'selfattentionembedding':
        kwargs = dict(
            output_size=args.emsize,
            num_units=args.self_attention_num_units,
            num_attention=args.self_attention_num_attention,
            dropout=args.self_attention_dropout,
            **kwargs,
        )
    elif name.lower() == 'lastoutputembedding':
        kwargs = dict(
            hidden_size=args.last_output_hidden_size,
            output_size=args.emsize,
            **kwargs,
        )
    elif name.lower() == 'wordprediction':
        kwargs = dict(
            hidden_size=args.wordprediction_hidden_size,
            activation=args.wordprediction_activation,
            **kwargs,
        )
    else:
        raise NotImplementedError(name)

    return create_(name, **kwargs)


def list_subwordnetworks(name=None):
    """List all registered networks."""
    reg = mx.registry.get_registry(SubwordNetwork)
    if not name:
        return list(reg.keys())
    else:
        return reg[name.lower()]


def alias(name):
    alias_ = mx.registry.get_alias_func(SubwordNetwork, 'metric')
    return alias_(name)


###############################################################################
# Subword encoders
###############################################################################
class SubwordNetwork(gluon.Block):
    """Models for sub-word embedding inference.

    Expects subword sequences in NTC layout.

    """

    min_size = 0


@register
@alias('fasttext')
# TODO(leezu): HybridBlock causes storage type fallback, but performs still
# better than non-sparse HybridBlock
class SumReduce(SubwordNetwork, gluon.HybridBlock):
    """Compute word embedding via summing over all subword embeddings."""

    def __init__(self, vocab_size, embed_size,
                 embedding_initializer=mx.init.Uniform(),
                 sparse_embedding=True, **kwargs):
        super(SumReduce, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.embedding_initializer = embedding_initializer
        self.sparse_embedding = sparse_embedding

        with self.name_scope():
            self.embedding = self._get_embedding()
            self.sum_reduce = gluon.nn.HybridLambda(
                lambda F, e, m: F.sum(
                    F.broadcast_mul(
                        e, F.expand_dims(m, axis=-1)),
                    axis=-2))

    def _get_embedding(self):
        if self.sparse_embedding:
            embedding = gluon.nn.SparseEmbedding(
                self.vocab_size, self.embed_size,
                weight_initializer=self.embedding_initializer)
        else:
            embedding = gluon.nn.Embedding(
                self.vocab_size, self.embed_size,
                weight_initializer=self.embedding_initializer)
        return embedding

    def hybrid_forward(self, F, inputs, mask, begin_state=None):
        """Defines the forward computation. Arguments can be either
        :py:class:`NDArray` or :py:class:`Symbol`."""
        subword_embeddings = self.embedding(inputs)
        out = self.sum_reduce(subword_embeddings, mask)
        return out


@register
# TODO(leezu): HybridBlock causes storage type fallback, but performs still
# better than non-sparse HybridBlock
class MeanReduce(SubwordNetwork, gluon.HybridBlock):
    """Compute word embedding via summing over all subword embeddings."""

    def __init__(self, vocab_size, embed_size,
                 embedding_initializer=mx.init.Uniform(),
                 sparse_embedding=True, **kwargs):
        super(MeanReduce, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.embedding_initializer = embedding_initializer
        self.sparse_embedding = sparse_embedding

        with self.name_scope():
            self.embedding = self._get_embedding()
            self.mean_reduce = gluon.nn.HybridLambda(
                lambda F, e, m: F.mean(
                    F.broadcast_mul(
                        e, F.expand_dims(m, axis=-1)),
                    axis=-2))

    def _get_embedding(self):
        if self.sparse_embedding:
            embedding = gluon.nn.SparseEmbedding(
                self.vocab_size, self.embed_size,
                weight_initializer=self.embedding_initializer)
        else:
            embedding = gluon.nn.Embedding(
                self.vocab_size, self.embed_size,
                weight_initializer=self.embedding_initializer)
        return embedding

    def hybrid_forward(self, F, inputs, mask, begin_state=None):
        """Defines the forward computation. Arguments can be either
        :py:class:`NDArray` or :py:class:`Symbol`."""
        subword_embeddings = self.embedding(inputs)
        out = self.mean_reduce(subword_embeddings, mask)
        return out


@register
class SubwordRNN(SubwordNetwork):
    """RNN model for sub-word embedding inference."""

    def __init__(self, mode, embed_size, hidden_size, output_size, num_layers,
                 embedding_dropout, bidirectional, encoder_dropout, vocab_size,
                 embedding_initializer=mx.init.Uniform(), **kwargs):
        super(SubwordRNN, self).__init__(**kwargs)
        # Embedding
        self.vocab_size = vocab_size
        self.embedding_dropout = embedding_dropout
        self.embedding_initializer = embedding_initializer

        # Encoder
        self.mode = mode
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.encoder_dropout = encoder_dropout

        # Output
        self.output_size = output_size

        with self.name_scope():
            self.embedding = self._get_embedding()
            self.encoder = self._get_encoder()

    def _get_embedding(self):
        embedding = gluon.nn.HybridSequential()
        with embedding.name_scope():
            embedding.add(
                gluon.nn.Embedding(
                    self.vocab_size, self.embed_size,
                    weight_initializer=self.embedding_initializer))
            if self.embedding_dropout:
                embedding.add(gluon.nn.Dropout(self.embedding_dropout))

            # Change NTC to TNC layout (for RNN)
            embedding.add(gluon.nn.HybridLambda(lambda F, x: x.swapaxes(0, 1)))
        return embedding

    def _get_encoder(self):
        if self.mode == 'rnn_relu':
            rnn_block = functools.partial(gluon.rnn.RNN, activation='relu')
        elif self.mode == 'rnn_tanh':
            rnn_block = functools.partial(gluon.rnn.RNN, activation='tanh')
        elif self.mode == 'lstm':
            rnn_block = gluon.rnn.LSTM
        elif self.mode == 'gru':
            rnn_block = gluon.rnn.GRU

        rnn = rnn_block(
            hidden_size=self.hidden_size, num_layers=self.num_layers,
            dropout=self.encoder_dropout, bidirectional=self.bidirectional,
            input_size=self.embed_size)
        return rnn

    def begin_state(self, *args, **kwargs):
        return self.encoder.begin_state(*args, **kwargs)

    def forward(self, inputs, mask, begin_state=None):
        """Defines the forward computation. Arguments can be either
        :py:class:`NDArray` or :py:class:`Symbol`."""
        F = mx.nd
        embeddings = self.embedding(inputs)

        # Switch mask from NT to TN
        mask_tn = F.transpose(mask)
        masked_embeddings = F.broadcast_mul(embeddings,
                                            F.expand_dims(mask_tn, axis=-1))

        if begin_state is None:
            begin_state = self.begin_state(batch_size=embeddings.shape[1],
                                           ctx=inputs.context)

        encoded, states = self.encoder(masked_embeddings, begin_state)
        return encoded


@register
class AWDRNN(SubwordNetwork):
    """Pretrained character language model with learned self attention.

    """

    def __init__(self, model, embed_size, hidden_size, num_layers, output_size,
                 vocab_size, awdrnn_path, **kwargs):
        super(AWDRNN, self).__init__(**kwargs)
        # AWDRNN
        self.model = model
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.awdrnn_path = awdrnn_path
        # Output
        self.output_size = output_size

        # Initialized outside of namescope, as no training needed
        self.awdrnn = self._get_awdrnn()

    def _get_awdrnn(self):
        return nlp.model.language_model.AWDRNN(
            self.model, self.vocab_size, self.embed_size, self.hidden_size,
            self.num_layers, True, 0, 0, 0, 0, 0)

    def initialize(self, *args, **kwargs):
        # Disable gradients for pretrained AWDRNN
        for name, param in self.awdrnn.collect_params().items():
            param.grad_req = 'null'

        # Normal initialization
        super(AWDRNN, self).initialize(*args, **kwargs)

        # Overwrite with pretrained parameters
        self.awdrnn.load_params(self.awdrnn_path)

    def begin_state(self, *args, **kwargs):
        return self.encoder.begin_state(*args, **kwargs)

    def forward(self, inputs, mask, begin_state=None):
        """Defines the forward computation. Arguments can be either
        :py:class:`NDArray` or :py:class:`Symbol`."""
        F = mx.nd

        with mx.autograd.pause():
            # Change NTC to TNC layout (for RNN)
            inputs = inputs.swapaxes(0, 1)
            mask_tn = F.transpose(mask)

            encoded = self.awdrnn.embedding(inputs)
            # Switch mask from NT to TN
            encoded = F.broadcast_mul(encoded, F.expand_dims(mask_tn, axis=-1))

            if not begin_state:
                begin_state = self.awdrnn.begin_state(
                    batch_size=inputs.shape[1], ctx=inputs.context)
            out_states = []
            for i, (e, s) in enumerate(zip(self.awdrnn.encoder, begin_state)):
                encoded, state = e(encoded, s)
                out_states.append(state)
        return encoded


@register
class SubwordCNN(SubwordNetwork, gluon.HybridBlock):
    """CNN model for sub-word embedding inference.

    Parameters
    ----------
    embed_size : int
        Dimension of embedding vectors for subword units.
    output_size : int
        Dimension of embedding vectors for word units.
    vocab_size : int, default 2**8
        Size of the input vocabulary. Usually the input vocabulary is the
        number of distinct bytes.
    """

    min_size = None  # Minimum length, corresponds to largest filter size

    def __init__(self, embed_size, output_size, filter_sizes,
                 embedding_dropout, cnn_dropout, vocab_size, **kwargs):
        super(SubwordCNN, self).__init__(**kwargs)
        self._embed_size = embed_size
        self._output_size = output_size
        self._vocab_size = vocab_size

        self.min_size = max(filter_sizes)

        self.num_feature_maps = embed_size
        self.filter_sizes = filter_sizes
        self.dropout_embedding = embedding_dropout
        self.dropout_cnn = cnn_dropout

        with self.name_scope():
            self.embedding = self._get_embedding()
            self.encoder = self._get_encoder()
            self.decoder = self._get_decoder()

    def _get_embedding(self):
        embedding = gluon.nn.HybridSequential()
        with embedding.name_scope():
            embedding.add(
                gluon.nn.Embedding(self._vocab_size, self._embed_size,
                                   weight_initializer=mx.init.Uniform(0.1)))

            # Change NTC to NCT layout (for CNN)
            embedding.add(gluon.nn.HybridLambda(lambda F, x: x.swapaxes(1, 2)))

            if self.dropout_embedding:
                embedding.add(gluon.nn.Dropout(self.dropout_embedding))
        return embedding

    def _get_encoder(self):
        encoder = gluon.nn.HybridSequential()
        with encoder.name_scope():
            # Concurrent Convolutions with different kernel sizes
            rec = gluon.contrib.nn.HybridConcurrent()
            with rec.name_scope():
                for size in self.filter_sizes:
                    seq = gluon.nn.HybridSequential()
                    with seq.name_scope():
                        seq.add(
                            mx.gluon.nn.Conv1D(channels=self.num_feature_maps,
                                               kernel_size=size, strides=1,
                                               use_bias=True, layout='NCW',
                                               activation='relu'))

                        seq.add(
                            gluon.nn.HybridLambda(
                                lambda F, x: F.max(x, axis=2)))
                        seq.add(gluon.nn.HybridLambda(
                            lambda F, x: F.reshape(x,
                                shape=[-1, self.num_feature_maps])))
                    rec.add(seq)
            encoder.add(rec)

            # Dropout
            if self.dropout_cnn:
                encoder.add(mx.gluon.nn.Dropout(self.dropout_cnn))

        return encoder

    def _get_decoder(self):
        output = gluon.nn.HybridSequential()
        with output.name_scope():
            output.add(gluon.nn.Dense(self._output_size, flatten=False))
        return output

    def hybrid_forward(self, F, inputs, mask):  # pylint: disable=arguments-differ
        """Defines the forward computation. Arguments can be either
        :py:class:`NDArray` or :py:class:`Symbol`."""

        embeddings = self.embedding(inputs)

        # Expand mask from NT to NCT where C is broadcasted
        embeddings_masked = F.broadcast_mul(embeddings, F.expand_dims(mask, 1))

        encoded = self.encoder(embeddings_masked)
        out = self.decoder(encoded)
        return out


@register
class SubwordVDCNN(SubwordNetwork, VDCNN):
    min_size = 8


###############################################################################
# Embedding and auxiliary models
###############################################################################
@register
class SelfAttentionEmbedding(SubwordNetwork, gluon.HybridBlock):
    """Predicts an embedding from a sequence."""

    def __init__(self, output_size, num_units, num_attention, dropout,
                 **kwargs):
        super(SelfAttentionEmbedding, self).__init__(**kwargs)

        self.num_units = num_units
        self.num_attention = num_attention
        self.dropout = dropout
        self.output_size = output_size

        with self.name_scope():
            self.rnn_to_rep = gluon.nn.Dense(self.output_size, flatten=False,
                                             activation='tanh')
            self.attention = self._get_self_attention()
            self.decoder = self._get_decoder()

    def _get_decoder(self):
        return gluon.nn.Dense(self.output_size, flatten=True)

    def _get_self_attention(self):
        return self_attention.StructuredSelfAttentionCell(
            units=self.num_units, num_attention=self.num_attention,
            dropout=self.dropout)

    def hybrid_forward(self, F, inputs, mask):
        """Defines the forward computation. Arguments can be either
        :py:class:`NDArray` or :py:class:`Symbol`."""
        # Switch outputs to NTC
        rep = self.rnn_to_rep(inputs)
        rep = F.swapaxes(rep, 0, 1)
        mask = F.expand_dims(mask, -2)
        mask = F.broadcast_axes(mask, 1, self.num_attention)
        context_vec, att_weights = self.attention(rep, mask)
        # Combine multiple attentions
        out = self.decoder(context_vec)
        return out, att_weights


@register
class LastOutputEmbedding(SubwordNetwork):
    """Predicts an embedding from a sequence."""

    def __init__(self, hidden_size, output_size, **kwargs):
        super(LastOutputEmbedding, self).__init__(**kwargs)

        self.hidden_size = hidden_size
        self.output_size = output_size

        with self.name_scope():
            self.rnn_to_rep = gluon.nn.Dense(self.hidden_size, flatten=False,
                                             activation='tanh')
            self.decoder = gluon.nn.Dense(
                self.output_size, in_units=self.hidden_size, flatten=True)

    def forward(self, inputs, last_valid):
        """Defines the forward computation. Arguments can be either
        :py:class:`NDArray` or :py:class:`Symbol`."""
        F = mx.nd

        enc = inputs[(last_valid, F.arange(inputs.shape[1],
                                           ctx=inputs.context))]
        rep = self.rnn_to_rep(enc)
        out = self.decoder(rep)
        return out


@register
class WordPrediction(SubwordNetwork, gluon.HybridBlock):
    """Predicts the word index from a sequence."""

    def __init__(self, hidden_size, vocab_size, activation, **kwargs):
        super(WordPrediction, self).__init__(**kwargs)

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.activation = activation

        with self.name_scope():
            self.hidden = gluon.nn.Dense(self.hidden_size, flatten=False,
                                         activation=self.activation)
            self.out = gluon.nn.Dense(self.vocab_size, flatten=False,
                                      in_units=self.hidden_size)

    def hybrid_forward(self, F, inputs):
        """Defines the forward computation. Arguments can be either
        :py:class:`NDArray` or :py:class:`Symbol`."""
        hidden = self.hidden(inputs)
        out = self.out(hidden)
        return out
