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

# pylint: disable=
"""Evaluation helpers for word embeddings."""

__all__ = [
    'WordEmbeddingSimilarityEvaluator',
    'WordEmbeddingNearestNeighborEvaluator', 'WordEmbeddingAnalogyEvaluator'
]

import random

import attr

import numpy as np
import mxnet as mx


@attr.s()
class WordEmbeddingEvaluator(object):
    dataset = attr.ib()
    vocabulary = attr.ib()


class WordEmbeddingSimilarityEvaluator(WordEmbeddingEvaluator):
    # Words and ground truth scores
    _w1s = None
    _w2s = None
    _scores = None
    _context = None

    def __attrs_post_init__(self):
        # Construct nd arrays from dataset
        w1s = []
        w2s = []
        scores = []
        for word1, word2, score in self.dataset:
            if (word1 in self.vocabulary and word2 in self.vocabulary):
                w1s.append(word1)
                w2s.append(word2)
                scores.append(score)

        print(("Using {num_use} of {num_total} word pairs "
               "from {ds} for evaluation.").format(
                   num_use=len(w1s),
                   num_total=len(self.dataset),
                   ds=self.dataset.__class__.__name__))

        self._w1s = w1s
        self._w2s = w2s
        self._scores = np.array(scores)

    def __len__(self):
        return len(self._w1s)

    def __call__(self, token_embedding):
        if not len(self):
            return 0

        w1s_embedding = mx.nd.L2Normalization(token_embedding(self._w1s))
        w2s_embedding = mx.nd.L2Normalization(token_embedding(self._w2s))

        import ipdb; ipdb.set_trace()
        batch_size, embedding_size = w1s_embedding.shape

        cosine_similarity = mx.nd.batch_dot(
            w1s_embedding.reshape((batch_size, 1, embedding_size)),
            w2s_embedding.reshape((batch_size, embedding_size, 1)))
        cosine_similarity_np = cosine_similarity.asnumpy().flatten()
        pearson_r = np.corrcoef(cosine_similarity_np, self._scores_np)[0, 1]
        return pearson_r


@attr.s()
class WordEmbeddingNearestNeighborEvaluator(WordEmbeddingEvaluator):
    num_base_words = attr.ib(default=5)
    num_nearest_neighbors = attr.ib(default=5)

    # Words and ground truth scores
    _words = None
    _indices = None

    def __attrs_post_init__(self):
        # Construct nd arrays from dataset
        self._words = []
        for word1, word2, score in self.dataset:
            for word in [word1, word2]:
                if word in self.token_embedding.token_to_idx:
                    self._words.append(words)
        random.shuffle(self._words)
        self._indices = mx.nd.array(
            [self.token_embedding.token_to_idx[w] for w in self._words],
            ctx=mx.cpu())

        print("Using " + str(self._words[:self.num_base_words]) +
              " as seeds for NN evaluation.")

    def __len__(self):
        return self._indices.shape[0]

    def __call__(self, embedding):
        words = self._indices.as_in_context(embedding.weight.list_ctx()[0])
        embedding = mx.nd.L2Normalization(embedding(words))

        similarity = mx.nd.dot(embedding, embedding.T).argsort(
            axis=1, is_ascend=0)

        eval_strs = []
        for i in range(self.num_nearest_neighbors):
            eval_strs.append(" ".join(
                words[int(idx.asscalar())]
                for idx in similarity[i][:self.num_nearest_neighbors]))
        return "\n".join(eval_strs)


@attr.s()
class WordEmbeddingAnalogyEvaluator(WordEmbeddingEvaluator):
    analogy = attr.ib(
        default="3CosMul",
        validator=attr.validators.in_(["3CosMul", "3CosAdd", "PairDirection"]))

    # Words and ground truth scores
    _w1s = None
    _w2s = None
    _scores = None

    def __attrs_post_init__(self):
        # Construct nd arrays from dataset
        w1s = []
        w2s = []
        scores = []
        for word1, word2, score in self.dataset:
            if (word1 in self.token_embedding.token_to_idx
                    and word2 in self.token_embedding.token_to_idx):
                w1s.append(self.token_embedding.token_to_idx[word1])
                w2s.append(self.token_embedding.token_to_idx[word2])
                scores.append(score)

        print(("Using {num_use} of {num_total} word pairs "
               "from {ds} for evaluation.").format(
                   num_use=len(w1s),
                   num_total=len(self.dataset),
                   ds=self.dataset.__class__.__name__))

        self._w1s = mx.nd.array(w1s, ctx=mx.cpu())
        self._w2s = mx.nd.array(w2s, ctx=mx.cpu())
        self._scores_np = np.array(scores)

    def __len__(self):
        return self._w1s.shape[0]

    def __call__(self, embedding):
        w1s = self._w1s.as_in_context(embedding.weight.list_ctx()[0])
        w2s = self._w2s.as_in_context(embedding.weight.list_ctx()[0])

        w1s_embedding = mx.nd.L2Normalization(embedding(w1s))
        w2s_embedding = mx.nd.L2Normalization(embedding(w2s))

        batch_size, embedding_size = w1s_embedding.shape

        cosine_similarity = mx.nd.batch_dot(
            w1s_embedding.reshape((batch_size, 1, embedding_size)),
            w2s_embedding.reshape((batch_size, embedding_size, 1)))
        cosine_similarity_np = cosine_similarity.asnumpy().flatten()
        pearson_r = np.corrcoef(cosine_similarity_np, self._scores_np)[0, 1]
        return pearson_r
