#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from math import ceil
from chainer import Chain, Function
import chainer.functions as F
import chainer.links as L
import chainer.utils as U
from chainer import reporter


class Similarity(Function):

    def __init__(self, beta):
        self.beta = beta

    def check_type_forward(self, in_types):
        U.type_check.expect(in_types.size() == 2)
        U.type_check.expect(
            in_types[0].dtype == np.float32,
            in_types[1].dtype == np.float32,
            in_types[0].shape == in_types[1].shape
        )

    def forward(self, inputs):
        x0, x1 = inputs
        n, c, h, w = x0.shape
        assert w == 1  # x0 and x1 must be vector @TODO: matrix
        self.diff = x0 - x1
        diff = self.diff
        distance = np.sum(diff * diff, axis=2)
        self.y = np.exp(- distance / (2 * self.beta)).reshape(n, c, 1, w)
        return self.y,

    def backward(self, inputs, gy):
        n, c, h, w = gy[0].shape
        coeff = - gy[0] * gy[0].dtype.type(self.y / self.beta)
        gx0 = coeff * self.diff
        return gx0, -gx0


def similarity(x0, x1, beta=1):
    return Similarity(beta)(x0, x1)


class KMaxPooling2D(F.pooling.pooling_2d.Pooling2D):

    def __init__(self, k):
        self.k = k

    def forward(self, x):
        img = x[0]
        n, c, h, w = img.shape
        indexes = np.sort(np.argsort(-img)[:, :, :, :self.k])
        indexes += np.arange(h).reshape((h, 1)) * w
        self.indexes = indexes
        y = np.take(x, indexes)
        return y,

    def backward(self, x, gy):
        n, c, out_h, out_w = gy[0].shape
        h, w = x[0].shape[2:]
        gcol = np.zeros((n * c * h * w), dtype=x[0].dtype)
        indexes = self.indexes.ravel()
        gcol[indexes] = gy[0].ravel()
        gx = gcol.reshape(n, c, h, w)
        return gx,


def k_max_pooling_2d(x, k):
    return KMaxPooling2D(k)(x)


class DynamicPooling2D(F.pooling.pooling_2d.Pooling2D):

    def __init__(self, kh, kw):
        self.kh = kh
        self.kw = kw

    def forward(self, x):
        img = x[0]
        n, c, h, w = img.shape
        kh, kw = self.kh, self.kw
        dh, rh = divmod(h, kh)
        dw, rw = divmod(w, kw)

        rows = self._split_rows(h, kh, dh, rh)
        blocks = self._split_cells(rows, w, kw, dw, rw)
        self.blocks = blocks

        # get max values from each block
        pool = np.zeros((n, c, kh, kw), dtype=img.dtype)
        indexes = np.zeros((n, c, kh, kw), dtype=np.int32)
        for i, row in enumerate(blocks):
            for j, cell in enumerate(row):
                block = img[:, :, cell[0][0]: cell[0][1], cell[1][0]: cell[1][1]]
                b_n, b_c, b_h, b_w = block.shape
                idxs = np.argsort(-block.reshape(b_n, b_c, b_h * b_w), axis=2)
                indexes[:, :, i:i + 1, j:j + 1] = idxs[:, :, 0:1].reshape(b_n, b_c, 1, 1)
                pool[:, :, i:i + 1, j:j + 1] = np.max(block, axis=(2, 3), keepdims=True)
        self.indexes = indexes
        return pool,

    @staticmethod
    def _split_rows(h, kh, dh, rh):
        rows = []
        if kh < h:
            if rh == 0:
                for i in range(dh):
                    rows.append((i * kh, (i + 1) * kh))
            else:
                for i in range(kh - rh):
                    rows.append((i * dh, (i + 1) * dh))
                s = (kh - rh) * dh
                for i in range(rh):
                    rows.append((s + i * (dh + 1), s + (i + 1) * (dh + 1)))
        else:
            for i in range(kh):
                d = i % h
                rows.append((d, d + 1))
        return rows

    @staticmethod
    def _split_cells(rows, w, kw, dw, rw):
        blocks = []
        if kw < w:
            if rw == 0:
                for row in rows:
                    cells = []
                    for i in range(dw):
                        cells.append((row, (i * kw, (i + 1) * kw)))
                    blocks.append(cells)
            else:
                for row in rows:
                    cells = []
                    for i in range(kw - rw):
                        cells.append((row, (i * dw, (i + 1) * dw)))
                    s = (kw - rw) * dw
                    for i in range(rw):
                        cells.append((row, (s + i * (dw + 1), s + (i + 1) * (dw + 1))))
                    blocks.append(cells)
        else:
            for row in rows:
                cells = []
                for i in range(kw):
                    d = i % w
                    cells.append((row, (d, d + 1)))
                blocks.append(cells)
        return blocks

    def backward(self, x, gy):
        n, c, out_h, out_w = gy[0].shape
        h, w = x[0].shape[2:]
        c_h, c_w = max(out_h, h), max(out_w, w)
        gcol = np.zeros((n * c * c_h * c_w), dtype=x[0].dtype)
        base = (np.arange(n) * (c * c_h * c_w)).reshape(n, 1, 1, 1) + (np.arange(c) * (c_h * c_w)).reshape(1, c, 1, 1)
        indexes = self.indexes

        base_h = 0
        for i, row in enumerate(self.blocks):
            b_h = row[0][0][1] - row[0][0][0]
            base_w = 0
            for j, cell in enumerate(row):
                b_w = cell[1][1] - cell[1][0]
                idxs_h = np.floor_divide(indexes[:, :, i:i + 1, j:j + 1], b_w) + base_h
                idxs_w = indexes[:, :, i:i + 1, j:j + 1] % b_w + base_w
                idxs = (base + idxs_h * c_w + idxs_w).ravel()
                gcol[idxs] = gy[0][:, :, i:i + 1, j:j + 1].ravel()
                base_w += b_w
            base_h += b_h
        gx = gcol.reshape(n, c, c_h, c_w)[:, :, :h, :w]
        return gx,


def dynamic_pooling_2d(x, kh, kw):
    return DynamicPooling2D(kh, kw)(x)


class BiCNN(Chain):
    """
    Yin, W. and Schütze, H., 2015, May. Convolutional neural network for paraphrase identification.
    In Proceedings of the 2015 Conference of the North American Chapter of the Association for
    Computational Linguistics: Human Language Technologies (pp. 901-911).
    https://aclweb.org/anthology/N/N15/N15-1091.pdf
    """

    def __init__(self, channels, filter_width, embeddings, max_sentence_length, k_top, beta, pool_size):
        vocab_size, embed_size = embeddings.shape
        maxlen = max_sentence_length
        assert maxlen % 2 == 0
        feature_size = [
            pool_size[0][0] * pool_size[0][1],
            pool_size[1][0] * pool_size[1][1] * channels[0],
            pool_size[2][0] * pool_size[2][1] * channels[1],
            pool_size[3][0] * pool_size[3][1] * channels[1],
        ]
        # initialize functions with parameters
        super(BiCNN, self).__init__(
            embed=L.EmbedID(
                in_size=vocab_size,
                out_size=embed_size,
                initialW=embeddings,
            ),
            # convolutions in the first block
            conv1l=self._create_convolution(
                in_channels=1,
                out_channels=channels[0],
                window_size=filter_width[0],
            ),
            conv1r=self._create_convolution(
                in_channels=1,
                out_channels=channels[0],
                window_size=filter_width[0],
            ),
            bias1l=L.Bias(axis=1, shape=(channels[0], ceil(embed_size / 2))),
            bias1r=L.Bias(axis=1, shape=(channels[0], ceil(embed_size / 2))),
            # convolutions in the first block
            conv2l=self._create_convolution(
                in_channels=channels[0],
                out_channels=channels[1],
                window_size=filter_width[1],
            ),
            conv2r=self._create_convolution(
                in_channels=channels[0],
                out_channels=channels[1],
                window_size=filter_width[1],
            ),
            bias2l=L.Bias(axis=1, shape=(channels[1], ceil(embed_size / 4))),
            bias2r=L.Bias(axis=1, shape=(channels[1], ceil(embed_size / 4))),
            # output layer
            linear=L.Linear(
                in_size=sum(feature_size),
                out_size=1,
                bias=0,
            ),
        )
        # retain parameters
        self._channels = channels
        self._filter_width = filter_width
        # self._feature_size = feature_size
        self._vocab_size = vocab_size
        self._embed_size = embed_size
        self._max_sentence_length = maxlen
        self._k_top = k_top
        self._beta = beta
        self._pool_size = pool_size

    def __call__(self, x):
        """
        forward computation
        """
        Sl, Sr = x[:, 0], x[:, 1]

        # first block
        El = self._to4d(F.transpose(self.embed(Sl), axes=(0, 2, 1)))  # each word in columns
        Er = self._to4d(F.transpose(self.embed(Sr), axes=(0, 2, 1)))  # each word in columns
        F_u = dynamic_pooling_2d(self.similarity(El, Er, self._beta), self._pool_size[0][0], self._pool_size[0][1])

        C1l = self.conv1l(El)
        C1r = self.conv1r(Er)
        A1l = self.folding(C1l)
        A1r = self.folding(C1r)
        F_sn = dynamic_pooling_2d(self.similarity(A1l, A1r, self._beta), self._pool_size[1][0], self._pool_size[1][1])

        B1l = F.tanh(self.bias1l(A1l))
        B1r = F.tanh(self.bias1r(A1r))
        pool1l = self.dynamic_kmax_pool(B1l, self._k_top, Sl.shape[1])
        pool1r = self.dynamic_kmax_pool(B1r, self._k_top, Sr.shape[1])

        # second block
        C2l = self.conv2l(pool1l)
        C2r = self.conv2r(pool1r)
        A2l = self.folding(C2l)
        A2r = self.folding(C2r)
        F_ln = dynamic_pooling_2d(self.similarity(A2l, A2r, self._beta), self._pool_size[2][0], self._pool_size[2][1])

        B2l = F.tanh(self.bias2l(A2l))
        B2r = F.tanh(self.bias2r(A2r))
        pool2l = k_max_pooling_2d(B2l, self._k_top)
        pool2r = k_max_pooling_2d(B2r, self._k_top)
        F_s = dynamic_pooling_2d(self.similarity(pool2l, pool2r, self._beta), self._pool_size[3][0], self._pool_size[3][1])

        # output
        y = self.linear(self._concat_features([F_u, F_sn, F_ln, F_s]))
        return F.reshape(y, (y.size, ))

    @staticmethod
    def _concat_features(Fs):
        f = []
        for F_mat in Fs:
            n, c, h, w = F_mat.shape
            f.append(F.reshape(F_mat, (n, c * h * w)))
        return F.concat(f, axis=1)

    @staticmethod
    def folding(x):
        x_odd = x[:, :, 1::2]  # extract odd rows
        x_even = x[:, :, ::2]  # extract even rows
        d = x_odd.shape[2] - x_even.shape[2]
        if d == -1:
            x_odd = BiCNN._add_row(x_odd)
        elif d == 1:
            x_even = BiCNN._add_row(x_even)
        return (x_odd + x_even) / 2

    @staticmethod
    def _add_row(x):
        n, c, w, h = x.shape
        return F.concat([np.zeros((n, c, 1, h), dtype=x.dtype), x], axis=2)

    @staticmethod
    def _to4d(x):
        n, h, w = x.shape
        return F.reshape(x, (n, 1, h, w))

    @staticmethod
    def similarity(x1, x2, beta):
        n, c, h1, w1 = x1.shape
        w2 = x2.shape[3]
        s = []
        for i in range(w1):
            for j in range(w2):
                s.append(similarity(x1[:, :, :, [i]], x2[:, :, :, [j]], beta))
        return F.reshape(F.concat(s, axis=2), (n, c, w1, w2))

    @staticmethod
    def dynamic_kmax_pool(x, k, size):
        k = int(max(k, size / 2 + 1))
        return k_max_pooling_2d(x, k)

    def _create_convolution(self, in_channels, out_channels, window_size):
        return L.Convolution2D(
            in_channels=in_channels,
            out_channels=out_channels,
            ksize=(1, window_size),  # filter size (height=1, width=m), which is equivalent m-grams
            stride=(1, 1),
            pad=(0, window_size - 1),  # this generates convolution matrices with dimension d * (|S| + m - 1)
            wscale=1,
            initialW=None,
            nobias=True,
            use_cudnn=True,
        )


class Classifier(Chain):
    compute_accuracy = True

    def __init__(self, predictor):
        assert isinstance(predictor, BiCNN)
        super(Classifier, self).__init__(predictor=predictor)
        self.lossfun = F.sigmoid_cross_entropy
        self.accfun = accuracy
        self.y = None
        self.loss = None
        self.accuracy = None

    def __call__(self, *args):
        """Computes the loss value for an input and label pair.

        It also computes accuracy and stores it to the attribute.

        Args:
            args (list of ~chainer.Variable): Input minibatch.

        The all elements of ``args`` but last one are features and
        the last element corresponds to ground truth labels.
        It feeds features to the predictor and compare the result
        with ground truth labels.

        Returns:
            ~chainer.Variable: Loss value.

        """

        assert len(args) >= 2
        x = args[:-1]
        t = args[-1]
        self.y = None
        self.loss = None
        self.accuracy = None
        self.y = self.predictor(*x)
        self.loss = self.lossfun(self.y, t)
        reporter.report({'loss': self.loss}, self)
        if self.compute_accuracy:
            self.accuracy = self.accfun(self.y, t)
            reporter.report({'accuracy': self.accuracy}, self)
        return self.loss


def accuracy(y, t):
    """Computes binary classification accuracy of the minibatch."""
    y = F.sigmoid(y)
    pred = np.array(y.data >= 0.5, dtype=t.dtype)
    return np.array((pred == t.data).mean(dtype=t.dtype)),
