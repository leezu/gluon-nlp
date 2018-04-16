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

from __future__ import print_function
import json
import mxnet as mx
import gluonnlp as nlp


def test_similarity():
    data = nlp.data.WordSim353(
        root='tests/data/wordsim353', segment="relatedness")
    token_embedding = nlp.embedding.create(
        "glove", source="glove.6B.50d.txt", unknown_token=None)

    evaluator = nlp.evaluation.WordEmbeddingSimilarityEvaluator()


if __name__ == '__main__':
    import nose
    nose.runmodule()
