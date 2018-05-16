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

import mxnet as mx
from mxnet import gluon

import gluonnlp as nlp
import self_attention


###############################################################################
# Hyperparameters
###############################################################################
def add_subword_parameters_to_parser(parser):
    group = parser.add_argument_group('Subword networks hyperparameters')
    group.add_argument('--subword-embedding-size', type=int, default=20,
                       help='Embedding size for each subword piece.')
    group.add_argument('--subword-embedding-dropout', type=float, default=0.0,
                       help='Embedding size for each subword piece.')

    _subwordrnn_args(parser)


def _subwordrnn_args(parser):
    group = parser.add_argument_group('SubwordRNN hyperparameters.')
    group.add_argument('--subwordrnn-mode', type=str, default='gru')
    group.add_argument('--subwordrnn-hidden-size', type=int, default=150)
    group.add_argument('--subwordrnn-num-layers', type=int, default=2)
    group.add_argument('--subwordrnn-encoder-dropout', type=float, default=0.0)
    group.add_argument('--subwordrnn-no-bidirectional', default=False,
                       action='store_true')
    group.add_argument('--subwordrnn-self-attention', default=False,
                       action='store_true',
                       help='If True, use self-attention on RNN states '
                       'to compute word embedding. '
                       'Otherwise use final states.')
    group.add_argument('--subwordrnn-self-attention-num-units', type=int,
                       default=350)
    group.add_argument('--subwordrnn-self-attention-num-attention', type=int,
                       default=10)
    group.add_argument('--subwordrnn-self-attention-dropout', type=float,
                       default=0.0)
    group.add_argument('--attention-regularizer-lambda', type=float,
                       default=1.0)


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

    # General arguments
    kwargs = dict(embed_size=args.subword_embedding_size,
                  embedding_dropout=args.subword_embedding_dropout,
                  output_size=args.emsize, **kwargs)

    # Network specific arguments
    if name.lower() == 'subwordrnn':
        kwargs = dict(
            mode=args.subwordrnn_mode, hidden_size=args.subwordrnn_hidden_size,
            num_layers=args.subwordrnn_num_layers,
            bidirectional=not args.subwordrnn_no_bidirectional,
            encoder_dropout=args.subwordrnn_encoder_dropout,
            use_self_attention=args.subwordrnn_self_attention,
            self_attention_num_units=args.subwordrnn_self_attention_num_units,
            self_attention_num_attention=args.
            subwordrnn_self_attention_num_attention,
            self_attention_dropout=args.subwordrnn_self_attention_dropout,
            **kwargs)
    else:
        raise NotImplementedError

    return create_(name, **kwargs)


def list_subwordnetworks(name=None):
    """List all registered networks."""
    reg = mx.registry.get_registry(SubwordNetwork)
    if not name:
        return list(reg.keys())
    else:
        return reg[name.lower()]


###############################################################################
# Model definitions
###############################################################################
class SubwordNetwork(gluon.Block):
    """Models for sub-word embedding inference.

    Expects subword sequences in NTC layout.

    """

    min_size = 0


@register
class SubwordRNN(SubwordNetwork):
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
    vocab_size : int, default 2**8
        Size of the input vocabulary. Usually the input vocabulary is the
        number of distinct bytes.
    dropout : float  # TODO
        Dropout rate to use for encoder output.
    use_self_attention : bool
        Use self attention to combine time-steps to a word embedding. Otherwise
        use final state.

    """

    def __init__(self, mode, embed_size, hidden_size, output_size, num_layers,
                 embedding_dropout, bidirectional, encoder_dropout, vocab_size,
                 use_self_attention, self_attention_num_units,
                 self_attention_num_attention, self_attention_dropout,
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

        self.use_self_attention = use_self_attention
        self.self_attention_num_units = self_attention_num_units
        self.self_attention_num_attention = self_attention_num_attention
        self.self_attention_dropout = self_attention_dropout

        # Output
        self.output_size = output_size

        with self.name_scope():
            self.embedding = self._get_embedding()
            self.encoder = self._get_encoder()
            self.rnn_to_rep = gluon.nn.Dense(self.output_size, flatten=False,
                                             activation='tanh')
            if self.use_self_attention:
                self.attention = self._get_self_attention()
            self.decoder = self._get_decoder()

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

    def _get_decoder(self):
        return gluon.nn.Dense(self.output_size, flatten=True)

    def _get_self_attention(self):
        return self_attention.StructuredSelfAttentionCell(
            units=self.self_attention_num_units,
            num_attention=self.self_attention_num_attention,
            dropout=self.self_attention_dropout)

    def begin_state(self, *args, **kwargs):
        return self.encoder.begin_state(*args, **kwargs)

    def forward(self, inputs, mask, last_valid, begin_state=None):
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

        if self.use_self_attention:
            # Switch outputs to NTC
            rep = self.rnn_to_rep(encoded)
            rep = rep.swapaxes(0, 1)
            attention_mask = mask.expand_dims(-2).broadcast_to(
                (mask.shape[0], self.self_attention_num_attention,
                 mask.shape[1]))
            context_vec, att_weights = self.attention(rep, attention_mask)
            # Combine multiple attentions
            out = self.decoder(context_vec)
            return out, att_weights
        else:
            assert len(states) == 1
            enc = encoded[last_valid,
                          mx.nd.arange(inputs.shape[0], ctx=inputs.context)]
            rep = self.rnn_to_rep(enc)
            out = self.decoder(rep)
            return out, None


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

    min_size = 5  # Minimum length, corresponds to largest filter size

    def __init__(self, embed_size, output_size, vocab_size=256, **kwargs):
        super(SubwordCNN, self).__init__(**kwargs)
        self._embed_size = embed_size
        self._output_size = output_size
        self._vocab_size = vocab_size

        self.num_feature_maps = embed_size
        self.filter_sizes = [5]
        self.dropout_embedding = 0.3
        self.dropout_cnn = 0.3

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

    def hybrid_forward(self, F, inputs, mask, begin_state=None):  # pylint: disable=arguments-differ
        """Defines the forward computation. Arguments can be either
        :py:class:`NDArray` or :py:class:`Symbol`."""

        embeddings = self.embedding(inputs)

        # Expand mask from NT to NCT where C is broadcasted
        embeddings_masked = F.broadcast_mul(embeddings, F.expand_dims(mask, 1))

        encoded = self.encoder(embeddings_masked)
        out = self.decoder(encoded)
        return out
