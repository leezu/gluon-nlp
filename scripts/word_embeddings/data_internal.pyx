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

# TODO cython: boundscheck=False
# TODO cython: wraparound=False
# TODO maybe cython: cdivision=True
# TODO maybe cython: embedsignature=True

"""Word embedding training datasets."""

import numpy as np
cimport numpy as cnp

from numpy cimport int64_t
# from libc.stdint cimport int64_t
from libcpp.vector cimport vector


cdef class SubwordLookup(object):
    """Just-in-time compiled helper class for fast subword lookup.

    Parameters
    ----------
    idx_to_subwordidxs list of list of int
         Mapping from vocabulary indices to a list of subword indices
    offset : int, default 0
         Offset to add on every subword index.

    """

    # cdef vector[int64_t[:]] idx_to_subwordidxs
    cdef list idx_to_subwordidxs
    cdef int offset

    def __cinit__(self, list idx_to_subwordidxs, int offset=0):
        # self.idx_to_subwordidxs = new vector[int64_t[:]]()
        self.idx_to_subwordidxs = new vector[int64_t[:]]()
        # self.idx_to_subwordidxs = []
        for subwords in idx_to_subwordidxs:
            self.idx_to_subwordidxs.push_back(np.array(subwords, dtype=np.int64))
        self.offset = offset

    cpdef skipgram(self, int64_t[:] indices):
        """Get a sparse COO array of words and subwords for SkipGram.

        Parameters
        ----------
        indices : iterable of int
            Array containing numbers in [0, vocabulary_size). The element
            at position idx is taken to be the word that occurs at row idx
            in the SkipGram batch.

        Returns
        -------
        numpy.ndarray of dtype float32
            Array containing weights such that for each row, all weights
            sum to 1. In particular, all elements in a row have weight 1 /
            num_elements_in_the_row
        numpy.ndarray of dtype int64
            This array is the row array of a sparse array of COO format.
        numpy.ndarray of dtype int64
            This array is the col array of a sparse array of COO format.

        """
        assert indices.ndim == 1

        cdef vector[int64_t] row
        cdef vector[int64_t] col
        cdef vector[float] data
        cdef int64_t idx
        with nogil:
            for i in range(indices.shape[0]):
                idx = indices[i]
                row.push_back(i)
                col.push_back(idx)
                data.push_back(1.0 / (1.0 + (<int64_t[:]>self.idx_to_subwordidxs[idx]).size))
                for subwordidxs in self.idx_to_subwordidxs[idx]:
                    row.push_back(i)
                    for subwordidx in subwordidxs:
                        col.push_back(subwordidx + self.offset)
                        data.push_back(1 / (1 + len(self.idx_to_subwordidxs[idx])))
        return (np.array(data, np.float32), np.array(row, dtype=np.int64),
                np.array(col, dtype=np.int64))

    def cbow(self, context_row, context_col):
        """Get a sparse COO array of words and subwords for CBOW.

        Parameters
        ----------
        context_row : numpy.ndarray of dtype int64
            Array of same length as context_col containing numbers in [0,
            batch_size). For each idx, context_row[idx] specifies the row
            that context_col[idx] occurs in a sparse matrix.
        context_col : numpy.ndarray of dtype int64
            Array of same length as context_row containing numbers in [0,
            vocabulary_size). For each idx, context_col[idx] is one of the
            context words in the context_row[idx] row of the batch.

        Returns
        -------
        numpy.ndarray of dtype float32
            Array containing weights summing to 1. The weights are chosen
            such that the sum of weights for all subwords and word units of
            a given context word is equal to 1 /
            number_of_context_words_in_the_row. This array is the data
            array of a sparse array of COO format.
        numpy.ndarray of dtype int64
            This array is the row array of a sparse array of COO format.
        numpy.ndarray of dtype int64
            This array is the col array of a sparse array of COO format.

        """
        row = []
        col = []
        data = []

        num_rows = np.max(context_row) + 1
        row_to_numwords = np.zeros(num_rows)

        for i, idx in enumerate(context_col):
            row_ = context_row[i]
            row_to_numwords[row_] += 1

            row.append(row_)
            col.append(idx)
            data.append(1 / (1 + len(self.idx_to_subwordidxs[idx])))
            for subword in self.idx_to_subwordidxs[idx]:
                row.append(row_)
                col.append(subword + self.offset)
                data.append(1 / (1 + len(self.idx_to_subwordidxs[idx])))

        # Normalize by number of words
        for i, row_ in enumerate(row):
            assert 0 <= row_ <= num_rows
            data[i] /= row_to_numwords[row_]

        return (np.array(data, np.float32), np.array(row, dtype=np.int64),
                np.array(col, dtype=np.int64))
