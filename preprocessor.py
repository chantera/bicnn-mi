#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from nltk.tokenize.punkt import PunktLanguageVars


class Tokenizer(PunktLanguageVars):

    def __init__(self):
        super(Tokenizer, self).__init__()
        pass

    def tokenize(self, document):
        return self.word_tokenize(document)


class Preprocessor:

    def __init__(self,
                 embed_file,
                 max_document_length,
                 unknown="<UNK>",
                 pad="<PAD>",
                 tokenizer=None):
        self._maxlen = max_document_length
        vocabulary, embeddings = self._load_embeddings(embed_file)
        embed_size = embeddings.shape[1]
        self._unknown = unknown
        self._pad = pad
        self._vocabulary = vocabulary
        self._embeddings = embeddings
        self._new_embeddings = []
        self._embed_size = embed_size
        if unknown not in vocabulary:
            self._add_vocabulary(unknown, random=False)
        if pad not in vocabulary:
            self._add_vocabulary(pad, random=False)
        if tokenizer:
            self._tokenizer = tokenizer
        else:
            self._tokenizer = Tokenizer()

    @staticmethod
    def _load_embeddings(path):
        vocabulary = {}
        embeddings = []
        with open(path) as f:
            for line in f:
                cols = line.strip().split(" ")
                word = cols[0]
                if word not in vocabulary:
                    vocabulary[word] = len(vocabulary)
                    embeddings.append(np.array(cols[1:], dtype=np.float32))
        return vocabulary, np.array(embeddings)

    def _add_vocabulary(self, word, random=True):
        # if word in self._vocabulary:
        #     return
        self._vocabulary[word] = len(self._vocabulary)
        if random:
            word_vector = np.random.uniform(-1, 1, self._embed_size)  # generate a random embedding for an unknown word
        else:
            word_vector = np.zeros(self._embed_size, dtype=np.float32)
        self._new_embeddings.append(word_vector)

    def fit(self, raw_documents):
        for document in raw_documents:
            self._fit_each(document)
        return self

    def _fit_each(self, raw_document):
        for token in self._tokenizer.tokenize(raw_document.lower()):
            if token not in self._vocabulary:
                self._add_vocabulary(token, random=True)
        return self

    def transform(self, raw_documents):
        samples = []
        for document in raw_documents:
            samples.append(self._transform_each(document))
        return np.array(samples, dtype=np.int32)

    def _transform_each(self, raw_document):
        tokens = self._tokenizer.tokenize(raw_document.lower())
        if len(tokens) > self._maxlen:
            print("Token length exceeds max_document_length")
            raise
        word_ids = np.full(self._maxlen, self._vocabulary[self._pad], dtype=np.int32)
        for i, token in enumerate(tokens):
            if token in self._vocabulary:
                word_ids[i] = self._vocabulary[token]
            else:
                word_ids[i] = self._vocabulary[self._unknown]
        return word_ids

    def fit_transform(self, raw_documents):
        return self.fit(raw_documents).transform(raw_documents)

    def _fit_transform_each(self, raw_document):
        return self._fit_each(raw_document)._transform_each(raw_document)

    def get_embeddings(self):
        if len(self._new_embeddings) > 0:
            self._embeddings = np.r_[self._embeddings, self._new_embeddings]
            self._new_embeddings = []
        return self._embeddings
