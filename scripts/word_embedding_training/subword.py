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

import mxnet as mx
from mxnet import gluon

import gluonnlp as nlp

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


def create(name, **kwargs):
    """Creates an instance of a registered network."""
    create_ = mx.registry.get_create_func(SubwordNetwork, 'subwordnetwork')
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
class SubwordRNN(SubwordNetwork, gluon.HybridBlock):
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

    """

    def __init__(self, mode, length, embed_size, hidden_size, output_size,
                 vocab_size=256, dropout=0.5, **kwargs):
        super(SubwordRNN, self).__init__(**kwargs)
        self._mode = mode
        self._length = length
        self._embed_size = embed_size
        self._hidden_size = hidden_size
        self._output_size = output_size
        self._dropout = dropout
        self._vocab_size = vocab_size

        self._bidirectional = True
        self._weight_dropout = 0
        self._var_drop_in = 0
        self._var_drop_state = 0
        self._var_drop_out = 0

        with self.name_scope():
            self.embedding = self._get_embedding()
            self.cell = self._get_cell(
                mode=self._mode, bidirectional=self._bidirectional,
                input_size=self._embed_size, hidden_size=self._hidden_size,
                weight_dropout=self._weight_dropout,
                var_drop_in=self._var_drop_in,
                var_drop_state=self._var_drop_state,
                var_drop_out=self._var_drop_out)
            self.decoder = self._get_decoder()

    def _get_embedding(self):
        embedding = gluon.nn.HybridSequential()
        with embedding.name_scope():
            embedding.add(
                gluon.nn.Embedding(self._vocab_size, self._embed_size,
                                   weight_initializer=mx.init.Uniform(0.1)))
            if self._dropout:
                embedding.add(gluon.nn.Dropout(self._dropout))

            # Change NTC to TNC layout (for RNN)
            embedding.add(gluon.nn.HybridLambda(lambda F, x: x.swapaxes(0, 1)))
        return embedding

    @staticmethod  # TODO waiting for mxnet RNN HybdridBlock support
    def _get_cell(*, mode, bidirectional, input_size, hidden_size,
                  weight_dropout, var_drop_in, var_drop_state, var_drop_out):
        cells = []  # Collect 1 or 2 cells depending on bidirectional
        for i in range(1 if not bidirectional else 2):
            if mode == 'rnn_relu':
                cell = gluon.rnn.RNNCell(hidden_size, 'relu',
                                         input_size=input_size)
            elif mode == 'rnn_tanh':
                cell = gluon.rnn.RNNCell(hidden_size, 'tanh',
                                         input_size=input_size)
            elif mode == 'lstm':
                cell = gluon.rnn.LSTMCell(hidden_size, input_size=input_size)
            elif mode == 'gru':
                cell = gluon.rnn.GRUCell(hidden_size, input_size=input_size)

            if var_drop_in + var_drop_state + var_drop_out != 0:
                cell = gluon.contrib.rnn.VariationalDropoutCell(
                    cell, var_drop_in, var_drop_state, var_drop_out)

            if weight_dropout:
                nlp.model.utils.apply_weight_drop(cell, 'h2h_weight',
                                                  rate=weight_dropout)

            cells.append(cell)

        if bidirectional:
            cell = gluon.rnn.BidirectionalCell(*cells)
        else:
            cell = cells[0]

        return cell

    def _get_decoder(self):
        output = gluon.nn.HybridSequential()
        with output.name_scope():
            output.add(gluon.nn.Dense(self._output_size, flatten=False))
        return output

    def begin_state(self, *args, **kwargs):
        return self.encoder.begin_state(*args, **kwargs)

    def hybrid_forward(self, F, inputs, mask, begin_state=None):  # pylint: disable=arguments-differ
        """Defines the forward computation. Arguments can be either
        :py:class:`NDArray` or :py:class:`Symbol`."""
        encoded = self.embedding(inputs)
        encoded, states = self.cell.unroll(length=self._length, inputs=encoded,
                                           begin_state=begin_state,
                                           merge_outputs=True, layout='TNC')
        if self._dropout:
            encoded = F.Dropout(encoded, p=self._dropout, axes=(0, ))
        out = self.decoder(encoded)

        # Switch mask from NT to TN
        mask = F.transpose(mask)
        out = F.broadcast_mul(out, F.expand_dims(mask, axis=-1))

        out = F.max(out, axis=0)
        return out


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
