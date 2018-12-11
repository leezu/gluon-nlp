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
"""Morphology datasets."""

import os
import random
import collections

import mxnet as mx
import gluonnlp as nlp

__all__ = ['LazaridouDerivationalMorphologyDataset']


class LazaridouDerivationalMorphologyDataset(mx.gluon.data.SimpleDataset):
    header = [
        'affix', 'stem', 'stemPOS', 'derived', 'derivedPOS', 'type',
        'relatedness_score', 'relatedness_class', 'quality_score',
        'quality_class']

    def __init__(self, segment='train', seed=0):
        """Lazaridou derivational morphology dataset

        Lazaridou et al., Compositional-ly derived representations of
        morphologically complex words in distributional semantics. ACL 2013

        Parameters
        ----------
        segment : ['train', 'test', 'all']
            Randomly sampled train and test set following the splitting
            procedure of Lazaridou.
        seed : int, default 1
            Seed used for random train, test split.

        """
        assert segment.lower() in ['train', 'test', 'all']
        segment = segment.lower()

        dirname = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(dirname, 'affix_complete_set.txt')

        data = nlp.data.TSVDataset(filename, num_discard_samples=1)

        if segment != 'all':
            affix_to_data = collections.defaultdict(list)
            for sample in data:
                affix_to_data[sample[0]].append(sample)

            rng = random.Random(seed)

            data = []
            for affix, affix_data in affix_to_data.items():
                rng.shuffle(affix_data)

                if segment == 'train':
                    data += affix_data[50:]
                elif segment == 'test':
                    data += affix_data[:50]
                else:
                    raise RuntimeError

        super(LazaridouDerivationalMorphologyDataset, self).__init__(
            list(data))

    @property
    def affixes(self):
        affixes = set(d[0] for d in self)
        return affixes
