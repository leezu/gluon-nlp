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


class SelfAttentionCell(gluon.HybridBlock):
    """Abstract self-attention cell
    """

    def __call__(self, value, mask=None):  # pylint: disable=arguments-differ
        """Compute the attention.

        Parameters
        ----------
        value : Symbol or NDArray or None, default None
            Key of the memory. Shape (batch_size, memory_length, key_dim)
            Value of the memory. Shape (batch_size, memory_length, value_dim)
        mask : Symbol or NDArray or None, default None
            Mask of the memory slots. Shape (batch_size, query_length, memory_length)
            Only contains 0 or 1 where 0 means that the memory slot will not be used.
            If set to None. No mask will be used.

        Returns
        -------
        context_vec : Symbol or NDArray
            Shape (batch_size, query_length, context_vec_dim)
        att_weights : Symbol or NDArray
            Attention weights. Shape (batch_size, query_length, memory_length)
        """
        return self.forward(value, mask)

    def hybrid_forward(self, F, value, mask=None):  # pylint: disable=arguments-differ
        att_weights = self._compute_weight(F, value, mask)
        context_vec = self._read_by_weight(F, att_weights, value)
        return context_vec, att_weights

    def forward(self, value, mask=None):  # pylint: disable=arguments-differ
        if mask is None:
            return super(SelfAttentionCell, self).forward(value)
        else:
            return super(SelfAttentionCell, self).forward(value, mask)

    def _read_by_weight(self, F, att_weights, value):
        """Read from the value matrix given the attention weights.

        Parameters
        ----------
        F : symbol or ndarray
        att_weights : Symbol or NDArray
            Attention weights.
            For single-head attention,
                Shape (batch_size, query_length, memory_length).
            For multi-head attention,
                Shape (batch_size, num_heads, query_length, memory_length).
        value : Symbol or NDArray
            Value of the memory. Shape (batch_size, memory_length, total_value_dim)

        Returns
        -------
        context_vec: Symbol or NDArray
            Shape (batch_size, query_length, context_vec_dim)
        """
        return F.batch_dot(att_weights, value)

    def _compute_weight(self, F, key, mask=None):
        """Compute attention weights based.

        Parameters
        ----------
        F : symbol or ndarray
        key : Symbol or NDArray
            Key of the memory. Shape (batch_size, memory_length, key_dim)
        mask : Symbol or NDArray or None
            Mask the memory slots. Shape (batch_size, query_length, memory_length)
            Only contains 0 or 1 where 0 means that the memory slot will not be used.
            If set to None. No mask will be used.

        Returns
        -------
        att_weights : Symbol or NDArray
            For single-head attention, Shape (batch_size, query_length, memory_length)
            For multi-head attentino, Shape (batch_size, num_heads, query_length, memory_length)
        """
        raise NotImplementedError


class StructuredSelfAttentionCell(SelfAttentionCell):
    """https://arxiv.org/pdf/1703.03130.pdf

    Parameters
    ----------
    units : int
        Hyperparameter that specifies the output of the Dense(query) layer.
    num_attention : int
        Number of attentions to apply.
    dropout : float, default 0.0
        Attention dropout.
    act : Activation, default nn.Activation('tanh')
    weight_initializer : str or `Initializer` or None, default None
        Initializer of the weights.
    bias_initializer : str or `Initializer`, default 'zeros'
        Initializer of the bias.
    prefix : str or None, default None
        See documentation of `Block`.
    params : ParameterDict or None, default None
        See documentation of `Block`.
    """

    def __init__(self, units, num_attention, dropout=0.0, act='tanh',
                 weight_initializer=None, prefix=None, params=None):
        super(StructuredSelfAttentionCell, self).__init__(
            prefix=prefix, params=params)
        self._units = units
        self._num_attention = num_attention
        self._act = act
        with self.name_scope():
            self._dropout_layer = gluon.nn.Dropout(dropout)
            self._value_map = gluon.nn.Dense(
                units=self._units, flatten=False, use_bias=False,
                activation=self._act, weight_initializer=weight_initializer,
                prefix='query_')
            self._attention_score = gluon.nn.Dense(
                units=self._num_attention, in_units=self._units, flatten=False,
                use_bias=False, weight_initializer=weight_initializer,
                prefix='score_')

    def _compute_weight(self, F, value, mask=None):
        mapped_value = self._value_map(value)
        att_score = self._attention_score(mapped_value)
        att_score = att_score.swapaxes(1, 2)
        # (batch_size, num_heads, memory_length)
        att_weights = self._dropout_layer(
            nlp.model.attention_cell._masked_softmax(F, att_score, mask))
        return att_weights
