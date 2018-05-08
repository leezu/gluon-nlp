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
"""Word embedding training datasetst."""

__all__ = ['Text8', 'SkipGramWordEmbeddingDataset']

import os
import shutil
import zipfile

import numba
import numpy as np
import numpy_indexed as npi
from mxnet.gluon.data.dataset import Dataset
from mxnet.gluon.utils import check_sha1, download

from .dataset import CorpusDataset


###############################################################################
# Preprocessing utilities
###############################################################################
@numba.njit(nogil=True)
def get_sentence_start_end(sentence_boundaries, sentence_idx):
    end = sentence_boundaries[sentence_idx]
    if sentence_idx == 0:
        start = 0
    else:
        start = sentence_boundaries[sentence_idx - 1]
    return start, end


@numba.njit(nogil=True)
def prune_sentences(coded, idx_to_pdiscard):
    '''Downsample frequent words.'''
    pruned = []
    for idx in coded:
        if np.random.uniform(0.0, 1.0) < idx_to_pdiscard[idx]:
            pruned.append(idx)
    return np.array(pruned, dtype=np.int32)


# TODO remove once numba 0.39 is released https://github.com/numba/numba/pull/2902
@numba.extending.overload(np.unique)
def np_unique(a):
    def np_unique_impl(a):
        b = np.sort(a.ravel())
        head = list(b[:1])
        tail = [x for i, x in enumerate(b[1:]) if b[i] != x]
        return np.array(head + tail)

    return np_unique_impl


###############################################################################
# Dataset / Samplers
###############################################################################
class _Hutter(CorpusDataset):
    def __init__(self, root, namespace, seq_len, bos, eos, pad):
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            os.makedirs(root)
        self._root = root
        self._base_url = 'http://mattmahoney.net/dc/'
        super(_Hutter, self).__init__(self._get_data())

    def _get_data(self):
        archive_file_name, archive_hash = self._archive_file
        data_file_name, data_hash = self._data_file[self._segment]
        root = self._root
        path = os.path.join(root, data_file_name)
        if not os.path.exists(path) or not check_sha1(path, data_hash):
            downloaded_file_path = download(self._base_url + archive_file_name,
                                            path=root, sha1_hash=archive_hash)

            with zipfile.ZipFile(downloaded_file_path, 'r') as zf:
                for member in zf.namelist():
                    filename = os.path.basename(member)
                    if filename:
                        dest = os.path.join(root, filename)
                        with zf.open(member) as source:
                            with open(dest, "wb") as target:
                                shutil.copyfileobj(source, target)
        return path


class Text8(_Hutter):
    def __init__(self, root=os.path.join('~', '.mxnet', 'datasets', 'text8'),
                 segment='train', seq_len=None, bos=None, eos=None, pad=None):
        self._archive_file = ('text8.zip',
                              '6c70299b93b7e1f927b42cd8f6ac1a31547c7a2e')
        self._data_file = {
            'train': ('text8', '0dc3edebc970dcc96137e7deda4d9995af9d93de')
        }
        self._segment = segment
        super(Text8, self).__init__(root, 'text8', seq_len, bos, eos, pad)


class _WordEmbeddingDataset(Dataset):
    """Dataset for word embedding training tasks.

    Takes a corpus dataset like Text8 as input. Processes sentences / lines in
    the dataset following the fasttext conventions. The implementation roughly
    follows the fasttext implementation but uses just in time compiled python
    code (with numba) instead of C++.

    Exposes batches for SkipGram or ContinousBagOfWord training.

    Parameters
    ----------
    sentences : Dataset
        Base dataset of sentences.
    window : int
        The size distance for training the embeddings. Considered to generate
        batches.
    negative : int
        Number of negative samples used for training. Currently > 0 is the only
        supported mode.
    min_count : int
        Discard words occuring less than min_count times in sentences dataset.
    sample : float
        Subsample frequent words.

    """

    def __init__(self, coded, idx_to_counts, idx_to_bytes,
                 fixed_size_subwords=True, window=5, negative=5, power=0.75):
        assert isinstance(idx_to_bytes, np.ndarray)

        self.idx_to_counts = idx_to_counts
        self.idx_to_bytes = idx_to_bytes
        self.window = window
        self.negative = negative
        self.power = power
        self.fixed_size_subwords = fixed_size_subwords

        # Flatten the datastructures
        self._sentence_boundaries = np.cumsum([len(s) for s in coded])
        self.coded = np.concatenate(coded)

        # Smoothed unigram counts for negative sampling. Negatives can be drawn
        # by sampling a number in [0, self._smoothed_cumsum[-1]) and finding
        # the respective index with np.searchsorted.
        self._smoothed_token_freq_cumsum = np.cumsum(
            (self.idx_to_counts**self.power).astype(np.int))

    def __getitem__(self, idx):
        # Separate Implementation for SkipGram and CBOW
        raise NotImplementedError

    def __len__(self):
        # 1 sample for every token over all sentences
        return self._sentence_boundaries[-1]

    @property
    def num_tokens(self):
        return len(self._smoothed_token_freq_cumsum)


class SkipGramWordEmbeddingDataset(_WordEmbeddingDataset):
    def __getitem__(self, idx):
        # Make sure idx is of shape (batch_size,)
        idx = np.array(idx).flatten()
        (source, target, label,
         unique_token_idxs, token_bytes) = _build_sg_batch(
             self.coded, idx, self.window, self.negative,
             self._smoothed_token_freq_cumsum, self._sentence_boundaries,
             self.idx_to_bytes, self.fixed_size_subwords)
        source_subword = npi.remap(
            source.flatten(), unique_token_idxs,
            np.arange(unique_token_idxs.shape[0])).reshape(source.shape)
        if len(idx) == 1:
            return (source[0], target[0], label[0], token_bytes,
                    source_subword[0])
        else:
            return (source, target, label, token_bytes, source_subword)


@numba.njit(nogil=True)
def _build_sg_batch(coded, idxs, window, negative, token_freq_cumsum,
                    sentence_boundaries, idx_to_bytes, fixed_size_subwords):
    batch_size = len(idxs)

    sources = np.zeros((batch_size, 1), np.float32)
    targets = np.zeros((batch_size, negative + 1), dtype=np.float32)
    labels = np.zeros((batch_size, negative + 1), dtype=np.float32)

    for i in numba.prange(batch_size):
        idx = idxs[i]
        source, target, label = _build_sg_item(coded, idx, window, negative,
                                               token_freq_cumsum,
                                               sentence_boundaries)
        sources[i] = source
        targets[i] = target
        labels[i] = label

    # Find the subword information for all tokens included in the batch
    unique_token_idxs_sources = np.unique(sources)
    unique_token_idxs_targets = np.unique(targets)
    unique_token_idxs = np.unique(
        np.concatenate((unique_token_idxs_sources, unique_token_idxs_targets)))
    token_bytes = idx_to_bytes[unique_token_idxs.astype(np.int32)]

    # Throw away unneeded padding zeros
    if not fixed_size_subwords:
        token_length = np.zeros((token_bytes.shape[0], ))
        for i in numba.prange(token_bytes.shape[0]):
            token_length[i] = np.argmax(token_bytes[i] == 0)
        token_bytes = token_bytes[:, :np.max(token_length)]

    return sources, targets, labels, unique_token_idxs, token_bytes


@numba.njit(nogil=True)
def _build_sg_item(coded, idx, window, negative, token_freq_cumsum,
                   sentence_boundaries):
    # TODO alternative to sampling, return all positive targets from fixed window size (to keep batch shapes identical)
    # To keep equivalence to word2vec, weight by distance to window center http://www.aclweb.org/anthology/Q15-1016

    sentence_idx = np.searchsorted(sentence_boundaries, idx)
    sentence_start, sentence_end = get_sentence_start_end(
        sentence_boundaries, sentence_idx)

    # TODO Throw away "sentences" for which assertion does not hold in Dataset class
    assert sentence_end - sentence_start > 1, "Can't sample positive target from sentence with only one word"

    # `b` in the original word2vec code
    random_reduced_window_size = np.random.randint(1, window)
    window_start_idx = max(sentence_start, idx - random_reduced_window_size)
    # First index outside of the window
    window_end_idx = min(sentence_end, idx + random_reduced_window_size + 1)

    # Sample positive_target_idx
    positive_target_idx = idx
    while positive_target_idx == idx:
        positive_target_idx = np.random.randint(window_start_idx,
                                                window_end_idx)

    # Prepare batch array
    source = np.full((1, ), coded[idx], np.float32)
    target = np.zeros((negative + 1, ), np.float32)
    label = np.zeros((negative + 1, ), np.float32)
    target[0] = coded[positive_target_idx]
    label[0] = 1

    # Sample negative
    for i in numba.prange(negative):
        target[i + 1] = np.searchsorted(token_freq_cumsum,
                                        np.random.randint(
                                            token_freq_cumsum[-1]))
        label[i + 1] = 0

    return source, target, label
