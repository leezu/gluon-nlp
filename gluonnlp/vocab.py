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

# pylint: disable=consider-iterating-dictionary
"""Vocabulary."""
from __future__ import absolute_import
from __future__ import print_function

__all__ = ['Vocab', 'SubwordVocab']

import sys
import json
import logging
import warnings
import collections
import itertools

from mxnet import nd, registry
import numpy as np
import numpy_indexed as npi

from .data.utils import DefaultLookupDict
from . import _constants as C
from . import embedding as emb


###############################################################################
# Token level vocabulary
###############################################################################
class Vocab(object):
    """Indexing and embedding attachment for text tokens.

    Parameters
    ----------
    counter : Counter or None, default None
        Counts text token frequencies in the text data. Its keys will be indexed according to
        frequency thresholds such as `max_size` and `min_freq`. Keys of `counter`,
        `unknown_token`, and values of `reserved_tokens` must be of the same hashable type.
        Examples: str, int, and tuple.
    max_size : None or int, default None
        The maximum possible number of the most frequent tokens in the keys of `counter` that can be
        indexed. Note that this argument does not count any token from `reserved_tokens`. Suppose
        that there are different keys of `counter` whose frequency are the same, if indexing all of
        them will exceed this argument value, such keys will be indexed one by one according to
        their __cmp__() order until the frequency threshold is met. If this argument is None or
        larger than its largest possible value restricted by `counter` and `reserved_tokens`, this
        argument has no effect.
    min_freq : int, default 1
        The minimum frequency required for a token in the keys of `counter` to be indexed.
    unknown_token : hashable object or None, default '<unk>'
        The representation for any unknown token. In other words, any unknown token will be indexed
        as the same representation. If None, looking up an unknown token will result in KeyError.
    padding_token : hashable object or None, default '<pad>'
        The representation for the special token of padding token.
    bos_token : hashable object or None, default '<bos>'
        The representation for the special token of beginning-of-sequence token.
    eos_token : hashable object or None, default '<eos>'
        The representation for the special token of end-of-sequence token.
    reserved_tokens : list of hashable objects or None, default None
        A list of reserved tokens (excluding `unknown_token`) that will always be indexed, such as
        special symbols representing padding, beginning of sentence, and end of sentence. It cannot
        contain `unknown_token` or duplicate reserved tokens. Keys of `counter`, `unknown_token`,
        and values of `reserved_tokens` must be of the same hashable type. Examples: str, int, and
        tuple.

    Attributes
    ----------
    embedding : instance of :class:`gluonnlp.embedding.TokenEmbedding`
        The embedding of the indexed tokens.
    idx_to_token : list of strs
        A list of indexed tokens where the list indices and the token indices are aligned.
    idx_to_counts : numpy.ndarray
        A list of the counts of tokens that were passed during Vocab construction.
    reserved_tokens : list of strs or None
        A list of reserved tokens that will always be indexed.
    token_to_idx : dict mapping str to int
        A dict mapping each token to its index integer.
    unknown_token : hashable object or None
        The representation for any unknown token. In other words, any unknown token will be indexed
        as the same representation.
    max_token_length : int, default None
        If not None, tokens for which len(token) > max_token_length is True are discarded.


    Examples
    --------

    >>> text_data = " hello world \\\\n hello nice world \\\\n hi world \\\\n"
    >>> counter = gluonnlp.data.count_tokens(text_data)
    >>> my_vocab = gluonnlp.Vocab(counter)
    >>> fasttext = gluonnlp.embedding.create('fasttext', source='wiki.simple.vec')
    >>> my_vocab.set_embedding(fasttext)
    >>> my_vocab.embedding[['hello', 'world']]
    [[  3.95669997e-01   2.14540005e-01  -3.53889987e-02  -2.42990002e-01
        ...
       -7.54180014e-01  -3.14429998e-01   2.40180008e-02  -7.61009976e-02]
     [  1.04440004e-01  -1.08580001e-01   2.72119999e-01   1.32990003e-01
        ...
       -3.73499990e-01   5.67310005e-02   5.60180008e-01   2.90190000e-02]]
    <NDArray 2x300 @cpu(0)>

    >>> my_vocab[['hello', 'world']]
    [5, 4]

    >>> input_dim, output_dim = my_vocab.embedding.idx_to_vec.shape
    >>> layer = gluon.nn.Embedding(input_dim, output_dim)
    >>> layer.initialize()
    >>> layer.weight.set_data(my_vocab.embedding.idx_to_vec)
    >>> layer(nd.array([5, 4]))
    [[  3.95669997e-01   2.14540005e-01  -3.53889987e-02  -2.42990002e-01
        ...
       -7.54180014e-01  -3.14429998e-01   2.40180008e-02  -7.61009976e-02]
     [  1.04440004e-01  -1.08580001e-01   2.72119999e-01   1.32990003e-01
        ...
       -3.73499990e-01   5.67310005e-02   5.60180008e-01   2.90190000e-02]]
    <NDArray 2x300 @cpu(0)>

    >>> glove = gluonnlp.embedding.create('glove', source='glove.6B.50d.txt')
    >>> my_vocab.set_embedding(glove)
    >>> my_vocab.embedding[['hello', 'world']]
    [[  -0.38497001  0.80092001
        ...
        0.048833    0.67203999]
     [  -0.41486001  0.71847999
        ...
       -0.37639001 -0.67541999]]
    <NDArray 2x50 @cpu(0)>

    """

    def __init__(self, counter=None, max_size=None, min_freq=1,
                 unknown_token=C.UNK_TOKEN, padding_token=C.PAD_TOKEN,
                 bos_token=C.BOS_TOKEN, eos_token=C.EOS_TOKEN,
                 reserved_tokens=None, max_token_length=None):

        # Sanity checks.
        assert min_freq > 0, '`min_freq` must be set to a positive value.'

        self._unknown_token = unknown_token
        special_tokens = []
        self._padding_token = padding_token
        if padding_token:
            special_tokens.append(padding_token)
        self._bos_token = bos_token
        if bos_token:
            special_tokens.append(bos_token)
        self._eos_token = eos_token
        if eos_token:
            special_tokens.append(eos_token)
        if reserved_tokens:
            special_tokens.extend(reserved_tokens)
            special_token_set = set(special_tokens)
            if unknown_token:
                assert unknown_token not in special_token_set, \
                    '`reserved_token` cannot contain `unknown_token`.'
            assert len(special_token_set) == len(special_tokens), \
                '`reserved_tokens` cannot contain duplicate reserved tokens or ' \
                'other special tokens.'
        self._index_special_tokens(unknown_token, special_tokens)
        self.max_token_length = max_token_length

        if counter:
            self._index_counter_keys(counter, unknown_token, special_tokens,
                                     max_size, min_freq)

        self._embedding = None

    def _index_special_tokens(self, unknown_token, special_tokens):
        """Indexes unknown and reserved tokens."""
        self._idx_to_token = [unknown_token] if unknown_token else []
        self._idx_to_counts = [0] if unknown_token else []

        if not special_tokens:
            self._reserved_tokens = None
        else:
            self._reserved_tokens = special_tokens[:]
            self._idx_to_token.extend(special_tokens)
            self._idx_to_counts.extend([-1] * len(special_tokens))

        if unknown_token is not None:
            self._token_to_idx = DefaultLookupDict(C.UNK_IDX)
        else:
            self._token_to_idx = {}
        self._token_to_idx.update(
            (token, idx) for idx, token in enumerate(self._idx_to_token))

    def _index_counter_keys(self, counter, unknown_token, special_tokens,
                            max_size, min_freq):
        """Indexes keys of `counter`.


        Indexes keys of `counter` according to frequency thresholds such as `max_size` and
        `min_freq`.
        """

        unknown_and_special_tokens = set(
            special_tokens) if special_tokens else set()

        if unknown_token is not None:
            unknown_and_special_tokens.add(unknown_token)

        token_freqs = sorted(counter.items(), key=lambda x: x[0])
        token_freqs.sort(key=lambda x: x[1], reverse=True)

        token_cap = len(unknown_and_special_tokens) + (len(counter)
                                                       if not max_size else
                                                       max_size)

        for token, freq in token_freqs:
            if (self.max_token_length is not None
                    and len(token) > self.max_token_length):
                continue
            if freq < min_freq or len(self._idx_to_token) == token_cap:
                break
            if token not in unknown_and_special_tokens:
                self._idx_to_token.append(token)
                self._idx_to_counts.append(freq)
                self._token_to_idx[token] = len(self._idx_to_token) - 1

    @property
    def embedding(self):
        return self._embedding

    @property
    def idx_to_token(self):
        return self._idx_to_token

    @property
    def idx_to_counts(self):
        return self._idx_to_counts

    @property
    def reserved_tokens(self):
        return self._reserved_tokens

    @property
    def token_to_idx(self):
        return self._token_to_idx

    @property
    def unknown_token(self):
        return self._unknown_token

    @property
    def padding_token(self):
        return self._padding_token

    @property
    def bos_token(self):
        return self._bos_token

    @property
    def eos_token(self):
        return self._eos_token

    def __contains__(self, token):
        """Checks whether a text token exists in the vocabulary.


        Parameters
        ----------
        token : str
            A text token.


        Returns
        -------
        bool
            Whether the text token exists in the vocabulary (including `unknown_token`).
        """

        return token in self._token_to_idx

    def __getitem__(self, tokens):
        """Looks up indices of text tokens according to the vocabulary.

        If `unknown_token` of the vocabulary is None, looking up unknown tokens results in KeyError.

        Parameters
        ----------
        tokens : str or list of strs
            A source token or tokens to be converted.


        Returns
        -------
        int or list of ints
            A token index or a list of token indices according to the vocabulary.
        """

        if not isinstance(tokens, (list, tuple)):
            return self._token_to_idx[tokens]
        else:
            return [self._token_to_idx[token] for token in tokens]

    def __len__(self):
        return len(self._idx_to_token)

    def set_embedding(self, *embeddings):
        """Attaches one or more embeddings to the indexed text tokens.


        Parameters
        ----------
        embeddings : None or tuple of :class:`gluonnlp.embedding.TokenEmbedding` instances
            The embedding to be attached to the indexed tokens. If a tuple of multiple embeddings
            are provided, their embedding vectors will be concatenated for the same token.
        """

        if len(embeddings) == 1 and embeddings[0] is None:
            self._embedding = None
            return

        for embs in embeddings:
            assert isinstance(embs, emb.TokenEmbedding), \
                'The argument `embeddings` must be an instance or a list of instances of ' \
                '`gluonnlp.embedding.TokenEmbedding`.'

        new_embedding = emb.TokenEmbedding(self.unknown_token)
        new_embedding._token_to_idx = self.token_to_idx
        new_embedding._idx_to_token = self.idx_to_token

        new_vec_len = sum(embs.idx_to_vec.shape[1] for embs in embeddings
                          if embs and embs.idx_to_vec is not None)
        new_idx_to_vec = nd.zeros(shape=(len(self), new_vec_len))

        col_start = 0
        # Concatenate all the embedding vectors in embedding.
        for embs in embeddings:
            if embs and embs.idx_to_vec is not None:
                col_end = col_start + embs.idx_to_vec.shape[1]
                # Cancatenate vectors of the unknown token.
                new_idx_to_vec[0, col_start:col_end] = embs[0]
                new_idx_to_vec[1:, col_start:col_end] = embs[
                    self._idx_to_token[1:]]
                col_start = col_end

        new_embedding._idx_to_vec = new_idx_to_vec
        self._embedding = new_embedding

    def to_tokens(self, indices):
        """Converts token indices to tokens according to the vocabulary.


        Parameters
        ----------
        indices : int or list of ints
            A source token index or token indices to be converted.


        Returns
        -------
        str or list of strs
            A token or a list of tokens according to the vocabulary.
        """

        to_reduce = False
        if not isinstance(indices, (list, tuple)):
            indices = [indices]
            to_reduce = True

        max_idx = len(self._idx_to_token) - 1

        tokens = []
        for idx in indices:
            if not isinstance(idx, int) or idx > max_idx:
                raise ValueError(
                    'Token index {} in the provided `indices` is invalid.'.
                    format(idx))
            else:
                tokens.append(self._idx_to_token[idx])

        return tokens[0] if to_reduce else tokens

    def to_indices(self, tokens):
        """Looks up indices of text tokens according to the vocabulary.


        Parameters
        ----------
        tokens : str or list of strs
            A source token or tokens to be converted.


        Returns
        -------
        int or list of ints
            A token index or a list of token indices according to the vocabulary.
        """

        return self[tokens]

    def __call__(self, tokens):
        """Looks up indices of text tokens according to the vocabulary.


        Parameters
        ----------
        tokens : str or list of strs
            A source token or tokens to be converted.


        Returns
        -------
        int or list of ints
            A token index or a list of token indices according to the vocabulary.
        """

        return self[tokens]

    def __repr__(self):
        return ('Vocab(size={}, unk="{}", reserved="{}", '
                'max_token_length)'.format(
                    len(self), self._unknown_token, self._reserved_tokens,
                    self.max_token_length))

    def to_json(self, path=None):
        """Serialize Vocab object to json string or write it to path.

        This method does not serialize the underlying embedding.
        """
        if self._embedding:
            warnings.warn('Serialization of attached embedding '
                          'to json is not supported. '
                          'You may serialize the embedding to a binary format '
                          'separately using vocab.embedding.serialize')
        vocab_dict = {}
        vocab_dict['idx_to_token'] = self._idx_to_token
        vocab_dict['idx_to_counts'] = self._idx_to_counts
        vocab_dict['token_to_idx'] = dict(self._token_to_idx)
        vocab_dict['reserved_tokens'] = self._reserved_tokens
        vocab_dict['unknown_token'] = self._unknown_token
        vocab_dict['padding_token'] = self._padding_token
        vocab_dict['bos_token'] = self._bos_token
        vocab_dict['eos_token'] = self._eos_token
        if path is None:
            return json.dumps(vocab_dict)
        else:
            with open(path, 'w') as f:
                json.dump(vocab_dict, f)

    @staticmethod
    def from_json(json_str):
        """Deserialize Vocab object from json string.

        Parameters
        ----------
        json_str : str
            Serialized json string of a Vocab object.


        Returns
        -------
        Vocab
        """
        vocab_dict = json.loads(json_str)

        unknown_token = vocab_dict.get('unknown_token')
        vocab = Vocab(unknown_token=unknown_token)
        vocab._idx_to_token = vocab_dict.get('idx_to_token')
        vocab._idx_to_counts = vocab_dict.get('idx_to_counts')
        vocab._token_to_idx = vocab_dict.get('token_to_idx')
        if unknown_token is not None:
            vocab._token_to_idx = DefaultLookupDict(
                vocab._token_to_idx[unknown_token], vocab._token_to_idx)
        vocab._reserved_tokens = vocab_dict.get('reserved_tokens')
        vocab._padding_token = vocab_dict.get('padding_token')
        vocab._bos_token = vocab_dict.get('bos_token')
        vocab._eos_token = vocab_dict.get('eos_token')
        return vocab


###############################################################################
# Subword level vocabulary
###############################################################################
class SubwordVocab(object):
    """Token index and string to subword unit mapping.

    Parameters
    ----------
    idx_to_token
        Known tokens for which the subword units should be precomputed.
    mode : str or list of ints
        Subword unit mode. If str, must be one of ['byte']. If list of ints,
        each integer specifies the n of ngrams to use.
    merge_indices : bool
        If True, subwords indices start from the num_tokens + 1 such that the
        embeddings for subword units and for tokens can be stored in the same
        matrix. If False, subword indices start from 0. merge_indices is
        ignored for 'byte' mode.

    """

    def __init__(self, idx_to_token, subword_function, merge_indices):
        self.idx_to_token = idx_to_token
        self.subword_function = subword_function
        self.merge_indices = merge_indices

        # Precompute a idx to subwordidxs mapping to support fast lookup
        self._idx_to_subwordidxs = list(subword_function(idx_to_token))
        logging.info(('Constructing subword vocabulary with {}. '
                      'The word with largest number of subwords '
                      'has {} subwords.').format(
                          subword_function,
                          max(len(s) for s in self._idx_to_subwordidxs)))

    def _do_handle_merge_indices(self, subwordindices):
        if self.merge_indices:
            subwordindices = [
                i + len(self.idx_to_token) for i in subwordindices
            ]
        return subwordindices

    def _undo_handle_merge_indices(self, subwordindices):
        if self.merge_indices:
            subwordindices = [
                i - len(self.idx_to_token) for i in subwordindices
            ]
        return subwordindices

    def indices_to_subwordindices(self, indices):
        subwordindices = [self._idx_to_subwordidxs[i] for i in indices]
        return self._do_handle_merge_indices(subwordindices)

    def words_to_subwordindices(self, words):
        subwordindices = self.subword_function(words)
        return self._do_handle_merge_indices(subwordindices)

    def subwordindices_to_subwords(self, subwordindices):
        subwordindices = self._undo_handle_merge_indices(subwordindices)
        return self.subword_function.indices_to_subwords(subwordindices)

    def subwords_to_subwordindices(self, subwords):
        return self.subword_function.subwords_to_subwordindices(subwords)

    def __len__(self):
        return len(self.subword_function)

    def to_json(self, path=None):
        """Serialize subword vocab object to json string or write it to path."""
        dict_ = {}
        try:
            dict_['subwordidx_to_subword'] = \
                self.subword_function.indices_to_subwords(
                    list(range(len(self.subword_function))))
        except RuntimeError:
            # Not all subword functions are invertible
            pass
        if path is None:
            return json.dumps(dict_)
        else:
            with open(path, 'w') as f:
                json.dump(dict_, f)


###############################################################################
# Subword functions and registry
###############################################################################
def register(subword_cls):
    """Registers a new subword function."""
    register_text_embedding = registry.get_register_func(
        _SubwordFunction, 'subword function')
    return register_text_embedding(subword_cls)


def create(subword_function_name, **kwargs):
    """Creates an instance of a subword function."""

    create_ = registry.get_create_func(_SubwordFunction, 'token embedding')
    return create_(subword_function_name, **kwargs)


def list_sources():
    """Get valid subword function names."""
    reg = registry.get_registry(_SubwordFunction)
    return list(reg.keys())


class _SubwordFunction(object):
    def __init__(self):
        pass

    def __call__(self, words):
        '''Return a generator over subwords in the given word.'''
        raise NotImplementedError

    def __len__(self):
        '''Return the number of subwords modeled.'''
        raise NotImplementedError

    def indices_to_subwords(self, indices):
        raise NotImplementedError

    def subwords_to_indices(self, subwords):
        raise NotImplementedError


@register
class ByteSubwords(_SubwordFunction):
    def __init__(self, encoding='utf-8'):
        self.encoding = encoding

    def __call__(self, words):
        generator = (np.frombuffer(word.encode(self.encoding),
                                   dtype=np.uint8).astype(np.int_)
                     for word in words)
        return generator

    def __len__(self):
        return 256

    def __repr__(self):
        return 'ByteSubwords(encoding={})'.format(self.encoding)

    def indices_to_subwords(self, indices):
        return indices

    def subwords_to_indices(self, subwords):
        return subwords


@register
class CharacterSubwords(_SubwordFunction):
    def __init__(self, vocabulary, min_freq=10):
        character_counter = collections.Counter(
            itertools.chain.from_iterable(vocabulary.idx_to_token))
        num_all_characters = len(character_counter)
        for character, count in character_counter.most_common():
            if count < min_freq:
                del character_counter[character]
        logging.info('Constructing subword vocabulary based on ngrams. '
                     f'Keeping {len(character_counter)} of '
                     f'{num_all_characters} characters.')
        assert sys.version_info >= (3, 6), 'Only Python 3.6+ supported. ' \
            'We rely on it\'s property of preserving '\
            'dictionary insertion order.'
        character_enumeration = enumerate(character_counter.most_common())

        self.subword_to_subwordidx = {
            w: i
            for i, (w, _) in character_enumeration
        }
        # Requires Py3.6+
        self.subwordidx_to_subword = list(self.subword_to_subwordidx.keys())

        # Information for __repr__
        self.vocabulary_repr = repr(vocabulary)

    def __call__(self, words):
        generator = (np.array([
            self.subword_to_subwordidx[c] for c in word
            if c in self.subword_to_subwordidx
        ]) for word in words)
        return generator

    def __len__(self):
        return len(self.subwordidx_to_subword)

    def __repr__(self):
        return ('CharacterSubwords(vocabulary={})'.format(
            self.vocabulary_repr))

    def indices_to_subwords(self, indices):
        return [self.subwordidx_to_subword[i] for i in indices]

    def subwords_to_indices(self, subwords):
        return [self.subword_to_subwordidx[w] for w in subwords]


@register
class NGramSubwords(_SubwordFunction):
    def __init__(self, vocabulary, max_num_subwords, ngrams=[3, 4, 5, 6]):
        self.ngrams = ngrams

        ngram_generator = self._get_all_ngram_generator(
            vocabulary.idx_to_token, self.ngrams)
        ngram_counter = collections.Counter(
            itertools.chain.from_iterable(ngram_generator))
        num_subwords = min(len(ngram_counter), max_num_subwords)
        logging.info('Constructing subword vocabulary based on ngrams. '
                     f'Keeping {num_subwords} of '
                     f'{len(ngram_counter)} subwords.')
        assert sys.version_info >= (3, 6), 'Only Python 3.6+ supported. ' \
            'We rely on it\'s property of preserving '\
            'dictionary insertion order.'
        subwords_enumeration = enumerate(
            ngram_counter.most_common(num_subwords))

        self.subword_to_subwordidx = {
            w: i
            for i, (w, _) in subwords_enumeration
        }
        # Requires Py3.6+
        self.subwordidx_to_subword = list(self.subword_to_subwordidx.keys())

        # Information for __repr__
        self.vocabulary_repr = repr(vocabulary)
        self.max_num_subwords = max_num_subwords
        self.ngrams = ngrams

    @staticmethod
    def _get_all_ngram_generator(words, ngrams):
        return ((('<' + word + '>')[i:i + N] for N in ngrams
                 for i in range((len(word) + 2) - N + 1)) for word in words)

    def __call__(self, words):
        generator = (np.array([
            self.subword_to_subwordidx[('<' + word + '>')[i:i + N]]
            for N in self.ngrams for i in range((len(word) + 2) - N + 1)
            if ('<' + word + '>')[i:i + N] in self.subword_to_subwordidx
        ]) for word in words)
        return generator

    def __len__(self):
        return len(self.subwordidx_to_subword)

    def __repr__(self):
        return ('NGramSubwords(vocabulary={}, '
                'max_num_subwords={}, ngrams={})'.format(
                    self.vocabulary_repr, self.max_num_subwords, self.ngrams))

    def indices_to_subwords(self, indices):
        return [self.subwordidx_to_subword[i] for i in indices]

    def subwords_to_indices(self, subwords):
        return [self.subword_to_subwordidx[w] for w in subwords]


@register
class NGramHashes(_SubwordFunction):
    def __init__(self, vocabulary, num_subwords, ngrams=[3, 4, 5, 6]):
        self.num_subwords = num_subwords
        self.ngrams = ngrams
        self.vocabulary = vocabulary

        # Information for __repr__
        self.vocabulary_repr = repr(vocabulary)
        self.ngrams = ngrams

    @staticmethod
    def fasttext_hash_asbytes(s, encoding='utf-8'):
        h = np.uint32(2166136261)
        s = s.encode(encoding)
        old_settings = np.seterr(all='ignore')
        for c in s:
            h = h ^ np.uint32(c)
            h = h * np.uint32(16777619)
        np.seterr(**old_settings)
        return h

    @staticmethod
    def _get_all_ngram_generator(words, ngrams):
        return ((('<' + word + '>')[i:i + N] for N in ngrams
                 for i in range((len(word) + 2) - N + 1)) for word in words)

    def __call__(self, words):
        generator = (np.array([
            self.fasttext_hash_asbytes(
                ('<' + word + '>')[i:i + N]) % self.num_subwords
            for N in self.ngrams for i in range((len(word) + 2) - N + 1)
        ]) for word in words)
        return generator

    def __len__(self):
        return self.num_subwords

    def __repr__(self):
        return ('NGramHashes(vocabulary={}, '
                'num_subwords={}, ngrams={})'.format(
                    self.vocabulary_repr, self.num_subwords, self.ngrams))

    def indices_to_subwords(self, indices):
        raise RuntimeError('ngram hash function is not invertible.')

    def subwords_to_indices(self, subwords):
        return [
            self.fasttext_hash_asbytes(sw) % self.num_subwords
            for sw in subwords
        ]
