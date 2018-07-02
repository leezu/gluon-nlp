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

# pylint: disable=abstract-method
"""Trainable embedding models."""

import gluonnlp as nlp

from mxnet import nd
from mxnet.gluon import nn

from gluonnlp.base import _str_types


class HighwayCNNEmbeddingModel(nlp.model.train.EmbeddingModel):
    subwordminlength = 6

    def __init__(self, token_to_idx, subword_function, embedding_size,
                 character_embedding_size, weight_initializer=None,
                 sparse_grad=True, **kwargs):
        super(HighwayCNNEmbeddingModel,
              self).__init__(embedding_size=embedding_size, **kwargs)
        self.token_to_idx = token_to_idx
        self.subword_function = subword_function
        self.weight_initializer = weight_initializer
        self.sparse_grad = sparse_grad
        self.character_embedding_size = character_embedding_size

        with self.name_scope():
            self.embedding = nn.Embedding(
                len(token_to_idx),
                embedding_size,
                weight_initializer=weight_initializer,
                sparse_grad=sparse_grad,
            )
            self.character_embedding = nn.Embedding(
                len(subword_function),
                character_embedding_size,
                weight_initializer=weight_initializer,
                sparse_grad=sparse_grad,
            )
            self.cnn = nlp.model.ConvolutionalEncoder(
                embed_size=character_embedding_size,
                output_size=self.embedding_size)

    def __contains__(self, token):
        # supports computing vector for any str that is at least either in the
        # word level vocabulary or contains subwords
        return (token in self.idx_to_token
                or self.subword_function([token])[0].shape[0])

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

        vecs = []

        ctx = self.embedding.weight.list_ctx()[0]
        for token in tokens:
            if token in self.token_to_idx:
                # Word is part of fastText model
                word = nd.array([self.token_to_idx[token]], ctx=ctx)
                wordmask = nd.ones_like(word)
            else:
                word = nd.array([0], ctx=ctx)
                wordmask = nd.zeros_like(word)
            subwords = self.subword_function([token])[0].expand_dims(0)

            # Enforce minlength
            if subwords.shape[1] < self.subwordminlength:
                new_subwords = nd.zeros((1, self.subwordminlength))
                subwordsmask = nd.zeros((1, self.subwordminlength), ctx=ctx)
                new_subwords[:, :subwords.shape[1]] = subwords.shape[1]
                subwordsmask[:, :subwords.shape[1]] = 1
                subwords = new_subwords
            else:
                subwordsmask = None

            subwords = subwords.as_in_context(ctx)
            if subwords.shape[1]:
                vec = self(word, subwords, wordsmask=wordmask,
                           subwordsmask=subwordsmask)
            else:
                # token is a special_token and subwords are not taken into account
                assert token in self.token_to_idx
                vec = self.embedding(word)

            vecs.append(vec)

        if squeeze:
            assert len(vecs) == 1
            return vecs[0].squeeze()
        else:
            return nd.concat(*vecs, dim=0)

    def __call__(self, words, subwords, wordsmask=None, subwordsmask=None,
                 words_to_unique_indices=None):
        return super(HighwayCNNEmbeddingModel, self).__call__(
            words, subwords, wordsmask, subwordsmask, words_to_unique_indices)

    def forward(self, words, characters, wordsmask=None, charactersmask=None,
                words_to_unique_indices=None):
        """Compute embedding of words in batch.

        Parameters
        ----------
        words : mx.nd.NDArray
            Array of token indices.
        characters : mx.nd.NDArray
            The characters associated with the tokens in `words`. If
            words_to_unique_indices is specified may contain the
            characters of the unique tokens in `words` with
            `words_to_unique_indices` containing the reverse mapping.
        wordsmask : mx.nd.NDArray, optional
            Mask for embeddings returend by the word level embedding operator.
        charactermask : mx.nd.NDArray, optional
            A mask for the subword embeddings looked up from `characters`.
            Applied before sum reducing the subword embeddings.
        words_to_unique_indices : mx.nd.NDArray, optional
            Mapping from the position in the `words` array to the position in
            the words_to_unique_characters_indices` array.

        """
        #pylint: disable=arguments-differ
        embeddings = self.embedding(words)
        if wordsmask is not None:
            wordsmask = nd.expand_dims(wordsmask, axis=-1)
            embeddings = nd.broadcast_mul(embeddings, wordsmask)
        else:
            wordsmask = 1

        # Swap axes for ConvolutionalEncoder input
        characters = characters.reshape((0, -1)).T
        if charactersmask is None:
            charactersmask = nd.ones_like(characters)
        charactersmask = charactersmask.reshape((0, -1)).T

        if words_to_unique_indices is None:
            # Characters batch_size in dim1
            assert words.shape[0] == characters.shape[1]
            character_embeddings = self.character_embedding(characters)
            subword_embeddings = self.cnn(character_embeddings, charactersmask)
            subword_embeddings = subword_embeddings.reshape(embeddings.shape)
            return (embeddings + subword_embeddings) / (wordsmask + 1)

        else:
            character_embeddings = self.character_embedding(characters)
            subword_embedding_weights = self.cnn(character_embeddings,
                                                 charactersmask)
            words_to_unique_indices = words_to_unique_indices.reshape(
                words.shape)
            subword_embeddings = nd.Embedding(
                data=words_to_unique_indices, weight=subword_embedding_weights,
                input_dim=subword_embedding_weights.shape[0],
                output_dim=self.embedding_size)
            subword_embeddings = subword_embeddings.reshape(embeddings.shape)
            return (embeddings + subword_embeddings) / (wordsmask + 1)
