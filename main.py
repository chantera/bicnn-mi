#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chainer import datasets, iterators, optimizers, training
from chainer.training import extensions as E
import numpy as np
from preprocessor import Preprocessor
from bicnn import BiCNN, Classifier
import os
import sys


class MsrpCorpusPreprocessor(Preprocessor):

    def __init__(self, embed_file):
        super(MsrpCorpusPreprocessor, self).__init__(
            embed_file=embed_file,
            max_document_length=48,
            unknown="*UNKNOWN*",
            pad="*PAD*",
        )

    # override
    def transform(self, X_raw):
        transform_each = super(MsrpCorpusPreprocessor, self)._transform_each
        X = []
        for X_raw_each in X_raw:
            X.append((transform_each(X_raw_each[0]), transform_each(X_raw_each[1])))
        return np.array(X)

    # override
    def fit_transform(self, X_raw):
        fit_transform_each = super(MsrpCorpusPreprocessor, self)._fit_transform_each
        X = []
        for X_raw_each in X_raw:
            X.append((fit_transform_each(X_raw_each[0]), fit_transform_each(X_raw_each[1])))
        return np.array(X)

    @property
    def max_sentence_length(self):
        return self._maxlen

    @property
    def embeddings(self):
        return self.get_embeddings()


def load_msrp_corpus(path):
    X = []
    y = []
    with open(path, encoding="utf_8_sig") as f:
        # next(f)  # skip header line
        for line in f:
            cols = line.strip().split("\t")  # Quality	#1 ID	#2 ID	#1 String	#2 String
            y.append(cols[0])
            X.append((cols[3], cols[4]))
    return X, np.array(y, dtype=np.int32)


def train(
        embed_file,
        train_file,
        test_file,
        n_epoch=20,
        batch_size=70):

    # Load files
    print("init preprocessor with %s" % embed_file)
    processor = MsrpCorpusPreprocessor(embed_file)
    print("load MSRParaphraseCorpus [train] from %s" % train_file)
    X_train_raw, y_train = load_msrp_corpus(train_file)
    print("load MSRParaphraseCorpus [test] from %s" % test_file)
    X_test_raw, y_test = load_msrp_corpus(test_file)

    print('')
    print("initialize ...")
    print('--------------------------------')
    print('# Minibatch-size: %d' % batch_size)
    print('# epoch: %d' % n_epoch)
    print('--------------------------------')

    # Preprocess data
    X_train = processor.fit_transform(X_train_raw)
    X_test = processor.transform(X_test_raw)

    # Set up a neural network to train
    model = Classifier(BiCNN(
        channels=[3, 5],
        filter_width=[6, 14],
        embeddings=processor.embeddings,
        max_sentence_length=processor.max_sentence_length,
        k_top=4,
        beta=2,
        pool_size=[(10, 10), (10, 10), (6, 6), (2, 2)]
    ))

    # Setup an optimizer
    optimizer = optimizers.AdaGrad(lr=0.01)
    optimizer.setup(model)

    # Initialize datasets
    train_iter = iterators.SerialIterator(datasets.TupleDataset(X_train, y_train), batch_size, repeat=True, shuffle=True)
    test_iter = iterators.SerialIterator(datasets.TupleDataset(X_test, y_test), batch_size, repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer)
    trainer = training.Trainer(updater, (n_epoch, 'epoch'), out='logs')

    # Set extensions
    trainer.extend(E.Evaluator(test_iter, model))
    trainer.extend(E.dump_graph('main/loss'))
    trainer.extend(E.snapshot(), trigger=(n_epoch, 'epoch'))
    trainer.extend(E.LogReport())
    trainer.extend(E.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy']))
    trainer.extend(E.ProgressBar(update_interval=2))

    # Run the training
    print("trainer.run() executed")
    print('')
    trainer.run()


def main(argv):
    basedir = os.path.dirname(os.path.realpath(__file__))
    sample = {
        'embed_file': basedir + "/sample/embeddings-original.EMBEDDING_SIZE=25.txt",
        'train_file': basedir + "/sample/msr_paraphrase_train-small.txt",
        'test_file': basedir + "/sample/msr_paraphrase_test-small.txt",
        'n_epoch': 20,
        'batch_size': 10
    }
    train(**sample)


if __name__ == "__main__":
    main(sys.argv)
