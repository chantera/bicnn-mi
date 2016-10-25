#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chainer import Chain
import chainer.functions as F
import chainer.links as L


class BCNN(Chain):
    """
    Yin, W., Schütze, H., Xiang, B. and Zhou, B., 2015.
    ABCNN: Attention-Based Convolutional Neural Network for Modeling Sentence Pairs.
    """

    def __init__(self, channels, filter_width, embeddings):
        vocab_size, embed_size = embeddings.shape
        super(BCNN, self).__init__(
            embed=L.EmbedID(
                in_size=vocab_size,
                out_size=embed_size,
                initialW=embeddings,
            ),
            conv=self._create_convolution(
                in_channels=1,
                out_channels=channels,
                window_size=filter_width,
            ),
            linear=L.Linear(
                in_size=embed_size * 2 * channels,
                out_size=1,
                bias=0,
            ),
        )

    def __call__(self, x):
        S0, S1 = x[:, 0], x[:, 1]
        E0 = self._to4d(F.transpose(self.embed(S0), axes=(0, 2, 1)))
        E1 = self._to4d(F.transpose(self.embed(S1), axes=(0, 2, 1)))
        C0 = self.conv(E0)
        C1 = self.conv(E1)
        A0 = F.average_pooling_2d(C0, ksize=(1, C0.shape[3]), stride=None, pad=0, use_cudnn=True)
        A1 = F.average_pooling_2d(C1, ksize=(1, C1.shape[3]), stride=None, pad=0, use_cudnn=True)

        y = self.linear(self._concat([A0, A1]))
        return F.reshape(y, (y.size, ))

    @staticmethod
    def _to4d(x):
        n, h, w = x.shape
        return F.reshape(x, (n, 1, h, w))

    @staticmethod
    def _concat(x):
        y = []
        for _x in x:
            n, c, h, w = _x.shape
            y.append(F.reshape(_x, (n, c * h * w)))
        return F.concat(y, axis=1)

    @staticmethod
    def _create_convolution(in_channels, out_channels, window_size):
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


class Classifier(L.Classifier):

    def __init__(self, predictor):
        assert isinstance(predictor, BCNN)
        super(Classifier, self).__init__(
            predictor=predictor,
            lossfun=F.sigmoid_cross_entropy,
            accfun=F.evaluation.binary_accuracy.binary_accuracy,
        )
