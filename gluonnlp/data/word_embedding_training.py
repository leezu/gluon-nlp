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

__all__ = [
    'Text8', 'Wikipedia', 'SkipGramWordEmbeddingDataset',
    'SkipGramFasttextWordEmbeddingDataset'
]

import io
import os
import shutil
import zipfile
import concurrent.futures
import functools
import pyarrow as pa
import pandas as pd

import numpy as np
import numpy_indexed as npi
from mxnet.gluon.data.dataset import Dataset
from mxnet.gluon.utils import check_sha1, download

import numba

from .dataset import CorpusDataset
from .utils import _get_home_dir, concat_sequence


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


@numba.njit(nogil=True)
def np_unique_wcounts(a):
    b = np.sort(a.flatten())
    unique = list(b[:1])
    counts = [1 for _ in unique]
    for x in b[1:]:
        if x != unique[-1]:
            unique.append(x)
            counts.append(1)
        else:
            counts[-1] += 1
    return np.array(unique), np.array(counts)


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
    def __init__(self, root=os.path.join(_get_home_dir(), 'datasets', 'text8'),
                 segment='train', seq_len=None, bos=None, eos=None, pad=None):
        self._archive_file = ('text8.zip',
                              '6c70299b93b7e1f927b42cd8f6ac1a31547c7a2e')
        self._data_file = {
            'train': ('text8', '0dc3edebc970dcc96137e7deda4d9995af9d93de')
        }
        self._segment = segment
        super(Text8, self).__init__(root, 'text8', seq_len, bos, eos, pad)


class Wikipedia(CorpusDataset):
    num_files = 100  # By convention we split the corpus to 100 files
    _s3_bucket = 'lllausen-data'
    _s3_key = 'datasets/wikimedia/{}/wiki.{}/{}'

    def __init__(self, date, language, root=os.path.join(
            _get_home_dir(), 'datasets', 'wikimedia'), i=None):
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            os.makedirs(root)
        self._root = root
        self.i = i
        self.date = date
        self.language = language
        super(Wikipedia, self).__init__(self._get_data())

    def _get_data(self):
        folder_name = 'wiki.{}'.format(self.language)
        data_file_name = format(self.i, '02d')
        root = self._root
        path = os.path.join(root, folder_name, data_file_name)

        # TODO(leezu): Publish the file hash together with the dataset on S3 and check
        if not os.path.exists(path):
            import boto3
            s3 = boto3.resource('s3')
            s3_key = self._s3_key.format(self.date, self.language, self.i)
            s3.Bucket(self._s3_bucket).download_file(s3_key, path)
        return path


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

    def __init__(self, coded, idx_to_counts, sentence_boundaries=None,
                 subword_vocab=None, keep_max_size=False, min_size=0, window=5,
                 negative=5, power=0.75):
        idx_to_subwordidxs = subword_vocab.indices_to_subwordindices(
            list(range(len(idx_to_counts))))

        # Convert variable length subword indices per token to a padded
        # numpy array. Pad with -1.
        max_subwordidxs_len = max(len(s) for s in idx_to_subwordidxs)
        idx_to_subwordidxs = np.stack(
            np.pad(b, (0, max_subwordidxs_len - len(b)
                       ), constant_values=-1, mode='constant')
            for b in idx_to_subwordidxs).astype(np.int)

        self.idx_to_counts = idx_to_counts
        self.idx_to_subwordidxs = idx_to_subwordidxs
        self.window = window
        self.negative = negative
        self.power = power
        self.keep_max_size = keep_max_size
        self.min_size = min_size

        # Flatten the datastructures
        if sentence_boundaries is None:
            # Throw away invalid sentences in the dataset
            coded = [c for c in coded if len(c) > 1]
            self._sentence_boundaries = np.cumsum([len(s) for s in coded])
            self.coded = np.concatenate(coded)
        else:
            self._sentence_boundaries = sentence_boundaries
            self.coded = coded

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


###############################################################################
# General Subword Embedding Training Dataset for SkipGram objective
###############################################################################
class SkipGramWordEmbeddingDataset(_WordEmbeddingDataset):
    def __getitem__(self, idx):
        # Make sure idx is of shape (batch_size,)
        idx = np.array(idx).flatten()
        (source, target, label, unique_sources_indices, unique_sources_counts,
         unique_sources_subwordsequences, unique_sources_subwordsequences_mask,
         unique_targets_indices, unique_targets_counts) = _build_sg_batch(
             self.coded, idx, self.window, self.negative,
             self._smoothed_token_freq_cumsum, self._sentence_boundaries,
             self.idx_to_subwordidxs, self.keep_max_size, self.min_size)
        source_subword = npi.remap(
            source.flatten(), unique_sources_indices,
            np.arange(unique_sources_indices.shape[0])).reshape(source.shape)
        if len(idx) == 1:
            return (source[0], target[0], label[0], unique_sources_indices,
                    unique_sources_counts, unique_sources_subwordsequences,
                    source_subword[0], unique_sources_subwordsequences_mask,
                    unique_targets_indices, unique_targets_counts)
        else:
            return (source, target, label, unique_sources_indices,
                    unique_sources_counts, unique_sources_subwordsequences,
                    source_subword, unique_sources_subwordsequences_mask,
                    unique_targets_indices, unique_targets_counts)


@numba.njit(nogil=True)
def _build_sg_batch(coded, idxs, window, negative, token_freq_cumsum,
                    sentence_boundaries, idx_to_subwordidxs, keep_max_size,
                    min_size):
    batch_size = len(idxs)

    num_sources = 1
    num_targets = negative + 1

    sources = np.zeros((batch_size, num_sources), dtype=np.float32)
    targets = np.zeros((batch_size, num_targets), dtype=np.float32)
    labels = np.zeros((batch_size, num_targets), dtype=np.float32)

    for i in numba.prange(batch_size):
        idx = idxs[i]
        source, target, label = _build_sg_item(coded, idx, window, negative,
                                               token_freq_cumsum,
                                               sentence_boundaries)
        sources[i] = source
        targets[i] = target
        labels[i] = label

    # Find the subword information for all sources and counts for all
    unique_sources_indices, unique_sources_counts = np_unique_wcounts(sources)
    unique_sources_subwordsequences = idx_to_subwordidxs[
        unique_sources_indices.astype(np.int32)]
    (unique_sources_subwordsequences,
     unique_sources_subwordsequences_mask) = _mask_2d(
         unique_sources_subwordsequences, keep_max_size, min_size)

    unique_targets_indices, unique_targets_counts = np_unique_wcounts(targets)

    return (sources, targets, labels, unique_sources_indices,
            unique_sources_counts, unique_sources_subwordsequences,
            unique_sources_subwordsequences_mask, unique_targets_indices,
            unique_targets_counts)


@numba.njit(nogil=True)
def _mask_2d(array, keep_max_size, min_size):
    assert len(array.shape) == 2
    token_length = np.zeros((array.shape[0], ))
    mask = np.zeros_like(array)
    for i in numba.prange(array.shape[0]):
        length = np.argmax(array[i] == -1)
        if length == 0:  # If -1 is not present
            length = array.shape[-1]
        token_length[i] = length
        array[i, length:] = 0
        mask[i, :length] = 1
    # Throw away unneeded padding zeros
    if not keep_max_size:
        new_length = max(np.max(token_length), min_size + 1)
        array = array[:, :new_length]
        mask = mask[:, :new_length]
    return array, mask


@numba.njit(nogil=True)
def _mask_3d(array, keep_max_size, min_size):
    assert len(array.shape) == 3
    token_length = np.zeros((array.shape[0] * array.shape[1], ))
    mask = np.zeros_like(array)
    for i in numba.prange(array.shape[0]):
        for j in range(array.shape[1]):
            length = np.argmax(array[i][j] == -1)
            if length == 0:  # If -1 is not present
                length = array.shape[-1]
            token_length[i * array.shape[1] + j] = length
            array[i, j, length:] = 0
            mask[i, j, :length] = 1
    # Throw away unneeded padding zeros
    if not keep_max_size:
        new_length = max(np.max(token_length), min_size + 1)
        array = array[:, :, :new_length]
        mask = mask[:, :, :new_length]
    return array, mask


@numba.njit(nogil=True)
def _build_sg_item(coded, idx, window, negative, token_freq_cumsum,
                   sentence_boundaries):
    # TODO alternative to sampling, return all positive targets from fixed window size (to keep batch shapes identical)
    # To keep equivalence to word2vec, weight by distance to window center http://www.aclweb.org/anthology/Q15-1016

    sentence_idx = np.searchsorted(sentence_boundaries, idx)
    sentence_start, sentence_end = get_sentence_start_end(
        sentence_boundaries, sentence_idx)

    assert sentence_end - sentence_start > 1, \
        "Can't sample positive target from sentence with only one word"

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


###############################################################################
# Fasttext Optimized Subword Embedding Training Dataset for SkipGram objective
###############################################################################
class SkipGramFasttextWordEmbeddingDataset(_WordEmbeddingDataset):
    def __getitem__(self, idx):
        # Make sure idx is of shape (batch_size,)
        idx = np.array(idx).flatten()
        (source, target, label, subword_mask) = _build_sg_fasttext_batch(
            self.coded, idx, self.window, self.negative,
            self._smoothed_token_freq_cumsum, self._sentence_boundaries,
            self.idx_to_subwordidxs, self.keep_max_size, self.min_size)
        if len(idx) == 1:
            return source[0], target[0], label[0], subword_mask[0]
        else:
            return source, target, label, subword_mask


@numba.njit(nogil=True)
def _build_sg_fasttext_batch(coded, idxs, window, negative, token_freq_cumsum,
                             sentence_boundaries, idx_to_subwordsequence,
                             keep_max_size, min_size):
    batch_size = len(idxs)
    # shape has +1 as fasttext also takes the token index itself
    max_subwordsequence_len = idx_to_subwordsequence.shape[1] + 1
    num_sources = 1
    num_targets = negative + 1

    sources = np.zeros((batch_size, num_sources, max_subwordsequence_len),
                       dtype=np.float32)
    targets = np.zeros((batch_size, num_targets), dtype=np.float32)
    labels = np.zeros((batch_size, num_targets), dtype=np.float32)

    for i in numba.prange(batch_size):
        idx = idxs[i]
        source, target, label = _build_sg_item(coded, idx, window, negative,
                                               token_freq_cumsum,
                                               sentence_boundaries)

        # Look up subword sequences for sources
        sources[i] = np.concatenate(
            (np.expand_dims(source, -1), idx_to_subwordsequence[source.astype(
                np.int32)]), axis=1)

        targets[i] = target
        labels[i] = label

    # Perform counts before -1 padding entries are replaced by 0
    _, unique_sources_counts = np_unique_wcounts(sources)
    unique_sources_counts = unique_sources_counts[1:]  # Ignore padding count
    _, unique_targets_counts = np_unique_wcounts(targets)

    sources, mask = _mask_3d(sources, keep_max_size, min_size)

    return (sources, targets, labels, mask, unique_sources_counts,
            unique_targets_counts)
