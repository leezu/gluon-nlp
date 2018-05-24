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
"""Word Embeddings Training
===========================

This example shows how to train word embeddings.

"""

import collections
import logging
import warnings

from mxboard import SummaryWriter
import mxnet as mx
from mxnet import gluon

import arguments
import evaluation
import gluonnlp as nlp
import utils
import subword
import word_embedding_training


def add_parameters(parser):
    group = parser.add_argument_group('Evaluation only specific settings')
    group.add_argument('path', type=str,
                       help='Path to pretrained TokenEmbedding file.')


def get_model(args):
    context = utils.get_context(args)
    embeddings_context = [context[0]]

    if '.bin' in args.path:
        # Assume binary fasttext format
        import gensim
        import struct
        import numpy as np

        gensim_fasttext = gensim.models.FastText()
        gensim_fasttext.file_name = args.path
        with open(args.path, 'rb') as f:
            gensim_fasttext._load_model_params(f)
            gensim_fasttext._load_dict(f)

            if gensim_fasttext.new_format:
                # quant input
                gensim_fasttext.struct_unpack(f, '@?')
            num_vectors, dim = gensim_fasttext.struct_unpack(f, '@2q')
            assert args.emsize == dim
            assert gensim_fasttext.wv.vector_size == dim
            dtype = np.float32 if struct.calcsize('@f') == 4 else np.float64
            matrix = np.fromfile(f, dtype=dtype, count=num_vectors * dim)
            matrix = matrix.reshape((-1, dim))

            num_words = len(gensim_fasttext.wv.vocab)
            num_subwords = gensim_fasttext.bucket
            assert num_words + num_subwords == num_vectors

        token_embedding = nlp.embedding.TokenEmbedding(unknown_token=None)
        token_embedding._idx_to_token = list(gensim_fasttext.wv.vocab.keys())
        token_embedding._idx_to_vec = mx.nd.array(matrix[:num_words])
        token_embedding._token_to_idx.update(
            (token, idx)
            for idx, token in enumerate(token_embedding._idx_to_token))
        vocab = nlp.Vocab(
            collections.Counter(token_embedding.idx_to_token),
            unknown_token=None, padding_token=None, bos_token=None,
            eos_token=None)
        vocab.set_embedding(token_embedding)

        # Get token embedding and subword token embedding
        embedding = gluon.nn.Embedding(num_words, args.emsize)
        embedding.initialize(ctx=embeddings_context)
        embedding.weight.set_data(vocab.embedding.idx_to_vec)

        if num_subwords:
            subword_function = nlp.vocab.create(
                'NGramHashes', vocabulary=vocab, num_subwords=num_subwords)
            subword_vocab = nlp.SubwordVocab(token_embedding.idx_to_token,
                                             subword_function=subword_function,
                                             merge_indices=False)

            if args.subword_network not in ['fasttext', 'sumreduce']:
                warnings.warn("Overwriting --subword-network to fasttext")
                args.subword_network = 'fasttext'
            subword_net = subword.create(name=args.subword_network, args=args,
                                         vocab_size=len(subword_vocab))
            subword_net.initialize(ctx=embeddings_context)
            subword_net.embedding.weight.set_data(
                mx.nd.array(matrix[num_words:]))
        else:
            subword_vocab = None
            subword_net = None
    else:
        token_embedding = nlp.embedding.TokenEmbedding.from_file(args.path)

        assert args.emsize == token_embedding.idx_to_vec.shape[1]
        vocab = nlp.Vocab(
            collections.Counter(token_embedding.idx_to_token),
            unknown_token=None, padding_token=None, bos_token=None,
            eos_token=None)
        vocab.set_embedding(token_embedding)

        # Output embeddings
        embedding = gluon.nn.Embedding(
            len(token_embedding.idx_to_token), args.emsize)
        embedding.initialize(ctx=embeddings_context)
        embedding.weight.set_data(vocab.embedding.idx_to_vec)

    return vocab, embedding, subword_vocab, subword_net


def load_and_evaluate(args):
    vocab, embedding_in, subword_vocab, subword_net = get_model(args)
    sw = SummaryWriter(logdir=args.logdir)
    sw.add_text(tag='args', text=str(args), global_step=0)
    eval_dict = evaluation.evaluate(args, embedding_in, subword_net, None,
                                    vocab, subword_vocab, sw)
    for k, v in eval_dict.items():
        for i in range(100):
            sw.add_scalar(tag=k, value=float(v), global_step=i)
            word_embedding_training.log(args, sw, embedding_in, None,
                                        subword_net, None, None, 0, 0, 0, 0, 0,
                                        None, i, vocab, subword_vocab)
    for k, v in eval_dict.items():
        print('{:.2f}'.format(v), '\t', k)


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    args_ = arguments.get_and_setup([add_parameters])
    load_and_evaluate(args_)
