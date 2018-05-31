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
"""Gluon Blocks
===============

"""

import mxnet as mx

from mxnet.gluon import nn, rnn
from mxnet.gluon.block import Block, HybridBlock


class ConvolutionBlock(HybridBlock):
    def __init__(self, channels, activation='relu', kernel_size=3,
                 layout='NCW', temporal_batchnorm=True,
                 weight_initializer=mx.initializer.MSRAPrelu(
                     factor_type='avg',
                     slope=0,
                 ), **kwargs):
        super(ConvolutionBlock, self).__init__(**kwargs)

        self.channels = channels
        self.activation = activation
        self.kernel_size = kernel_size
        self.layout = layout
        self.temporal_batchnorm = temporal_batchnorm
        self.weight_initializer = weight_initializer

        with self.name_scope():
            self.conv1 = nn.Conv1D(channels=channels, kernel_size=kernel_size,
                                   padding=1, use_bias=True,
                                   layout=self.layout,
                                   weight_initializer=self.weight_initializer)
            self.conv2 = nn.Conv1D(channels=channels, kernel_size=kernel_size,
                                   padding=1, use_bias=True,
                                   layout=self.layout,
                                   weight_initializer=self.weight_initializer)

            if self.temporal_batchnorm:
                self.norm1 = nn.BatchNorm(axis=self.layout.find('C'))
                self.norm2 = nn.BatchNorm(axis=self.layout.find('C'))

            self.activation = nn.Activation(self.activation)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        if self.temporal_batchnorm:
            x = self.norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        if self.temporal_batchnorm:
            x = self.norm2(x)
        x = self.activation(x)
        return x


class VDCNN(HybridBlock):
    def __init__(
            self, vocab_size, embed_size, depth, output_size,
            embedding_dropout=False, skip_connections=False, activation='relu',
            kernel_size=3, layout='NCW', temporal_batchnorm=True,
            depth_to_num_conv_block={
                9: [2, 2, 2, 2],
                17: [4, 4, 4, 4],
                29: [10, 10, 4, 4],
                49: [16, 16, 10, 4],
            }, conv_block_to_channels=[64, 128, 256, 512],
            convolution_initializer=mx.initializer.MSRAPrelu(
                factor_type='avg',
                slope=0,
            ), **kwargs):
        super(VDCNN, self).__init__(**kwargs)
        assert depth in depth_to_num_conv_block.keys()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.depth = depth
        self.output_size = output_size
        self.embedding_dropout = embedding_dropout
        self.skip_connections = skip_connections
        self.activation = activation
        self.kernel_size = kernel_size
        self.layout = layout
        self.temporal_batchnorm = temporal_batchnorm
        self.conv_block_to_channels = conv_block_to_channels
        self.depth_to_num_conv_block = depth_to_num_conv_block
        self.convolution_initializer = convolution_initializer

        with self.name_scope():
            self.embedding = self._get_embedding()
            self.encoder = self._get_encoder()
            self.decoder = self._get_decoder()

    def _get_embedding(self):
        embedding = nn.HybridSequential()
        with embedding.name_scope():
            embedding.add(
                nn.Embedding(self.vocab_size, self.embed_size,
                             weight_initializer=mx.init.Uniform(0.1)))
            # Change NTC to NCT layout (for CNN)
            embedding.add(nn.HybridLambda(lambda F, x: x.swapaxes(1, 2)))
            if self.embedding_dropout:
                embedding.add(nn.Dropout(self.embedding_dropout))
        return embedding

    def _get_encoder(self):
        encoder = nn.HybridSequential()
        with encoder.name_scope():
            # First convolution layer
            encoder.add(
                nn.Conv1D(
                    channels=self.conv_block_to_channels[0],
                    kernel_size=self.kernel_size,
                    padding=1,
                    layout=self.layout,
                    weight_initializer=self.convolution_initializer,
                ))

            # Convolution blocks
            if not self.skip_connections:
                for i, channels in enumerate(self.conv_block_to_channels):
                    num_blocks = self.depth_to_num_conv_block[self.depth][i]
                    # num_blocks ConvolutionBlocks
                    for j in range(num_blocks):
                        encoder.add(
                            ConvolutionBlock(
                                channels=channels,
                                activation=self.activation,
                                kernel_size=self.kernel_size,
                                layout=self.layout,
                                temporal_batchnorm=self.temporal_batchnorm,
                                weight_initializer=self.
                                convolution_initializer,
                            ))

                    # Pooling
                    encoder.add(
                        nn.MaxPool1D(
                            pool_size=self.kernel_size,
                            padding=1,
                            strides=2,
                        ))

            else:
                raise NotImplementedError

            # k-max-pooling (but we use max pooling here) The reason is that
            # k-maxpooling would require a minimum sequence length of k*8 and
            # many words are short
            encoder.add(nn.HybridLambda(lambda F, x: F.max(x, axis=2)))

        return encoder

    def _get_decoder(self):
        decoder = nn.HybridSequential()
        with decoder.name_scope():
            # Dense layers
            decoder.add(
                nn.Dense(
                    units=self.conv_block_to_channels[0],  # * k, k==1
                    activation=self.activation,
                ))
            decoder.add(
                nn.Dense(
                    units=self.conv_block_to_channels[0],
                    activation=self.activation,
                ))
            decoder.add(nn.Dense(units=self.output_size))

        return decoder

    def hybrid_forward(self, F, x, mask):
        x = self.embedding(x)
        # Expand mask from NT to NCT where C is broadcasted
        x = F.broadcast_mul(x, F.expand_dims(mask, 1))
        x = self.encoder(x)
        x = self.decoder(x)
        return x
