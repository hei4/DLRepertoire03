# -*- coding: utf-8 -*-
import numpy

import chainer
from chainer.backends import cuda
import chainer.functions as F
import chainer.links as L


def add_noise(h, sigma=0.2):
    xp = cuda.get_array_module(h.data)
    if chainer.config.train:
        return h + sigma * xp.random.randn(*h.shape)
    else:
        return h


class Generator(chainer.Chain):

    def __init__(self, n_hidden, image_size=64, ch=512, wscale=0.02):
        super(Generator, self).__init__()
        self.n_hidden = n_hidden
        self.ch = ch
        self.image_size = image_size
        self.bottom_size = image_size // 16

        with self.init_scope():
            w = chainer.initializers.Normal(wscale)
            
            self.l0 = L.Linear(None, self.bottom_size * self.bottom_size * ch, initialW=w)
            self.dc1 = L.Deconvolution2D(None, ch // 2, 4, 2, 1, initialW=w)
            self.dc2 = L.Deconvolution2D(None, ch // 4, 4, 2, 1, initialW=w)
            self.dc3 = L.Deconvolution2D(None, ch // 8, 4, 2, 1, initialW=w)
            self.dc4 = L.Deconvolution2D(None, 3,       4, 2, 1, initialW=w) # L.Deconvolution2D(None, 3, 3, 1, 1, initialW=w)
            self.bn0 = L.BatchNormalization(self.bottom_size * self.bottom_size * ch)
            self.bn1 = L.BatchNormalization(ch // 2)
            self.bn2 = L.BatchNormalization(ch // 4)
            self.bn3 = L.BatchNormalization(ch // 8)

    def make_hidden(self, batchsize):
        return numpy.random.normal(0, 1., (batchsize, self.n_hidden, 1, 1)).astype(numpy.float32)
        # return numpy.random.uniform(-1, 1, (batchsize, self.n_hidden, 1, 1)).astype(numpy.float32)

    def __call__(self, z):
        h = F.reshape(
            F.leaky_relu(self.bn0(self.l0(z))), # F.relu(self.bn0(self.l0(z))),
            (len(z), self.ch, self.bottom_size, self.bottom_size)
        )
        h = F.leaky_relu(self.bn1(self.dc1(h))) # F.relu(self.bn1(self.dc1(h)))
        h = F.leaky_relu(self.bn2(self.dc2(h))) # F.relu(self.bn2(self.dc2(h)))
        h = F.leaky_relu(self.bn3(self.dc3(h))) # F.relu(self.bn3(self.dc3(h)))
        x = F.tanh(self.dc4(h)) # x = F.sigmoid(self.dc4(h))
        return x


class Discriminator(chainer.Chain):

    def __init__(self, bottom_width=4, ch=512, wscale=0.02):
        w = chainer.initializers.Normal(wscale)
        super(Discriminator, self).__init__()
        with self.init_scope():
            self.c0_0 = L.Convolution2D(None, ch // 8, 3, 1, 1, initialW=w)
            self.c0_1 = L.Convolution2D(None, ch // 4, 4, 2, 1, initialW=w)
            self.c1_0 = L.Convolution2D(None, ch // 4, 3, 1, 1, initialW=w)
            self.c1_1 = L.Convolution2D(None, ch // 2, 4, 2, 1, initialW=w)
            self.c2_0 = L.Convolution2D(None, ch // 2, 3, 1, 1, initialW=w)
            self.c2_1 = L.Convolution2D(None, ch // 1, 4, 2, 1, initialW=w)
            self.c3_0 = L.Convolution2D(None, ch // 1, 3, 1, 1, initialW=w)
            self.l4 = L.Linear(None, 1, initialW=w)
            self.bn0_1 = L.BatchNormalization(ch // 4, use_gamma=False)
            self.bn1_0 = L.BatchNormalization(ch // 4, use_gamma=False)
            self.bn1_1 = L.BatchNormalization(ch // 2, use_gamma=False)
            self.bn2_0 = L.BatchNormalization(ch // 2, use_gamma=False)
            self.bn2_1 = L.BatchNormalization(ch // 1, use_gamma=False)
            self.bn3_0 = L.BatchNormalization(ch // 1, use_gamma=False)

    def __call__(self, x):
        h = add_noise(x)
        h = F.leaky_relu(add_noise(self.c0_0(h)))
        h = F.leaky_relu(add_noise(self.bn0_1(self.c0_1(h))))
        h = F.leaky_relu(add_noise(self.bn1_0(self.c1_0(h))))
        h = F.leaky_relu(add_noise(self.bn1_1(self.c1_1(h))))
        h = F.leaky_relu(add_noise(self.bn2_0(self.c2_0(h))))
        h = F.leaky_relu(add_noise(self.bn2_1(self.c2_1(h))))
        h = F.leaky_relu(add_noise(self.bn3_0(self.c3_0(h))))
        return self.l4(h)


class Discriminator2(chainer.Chain):

    def __init__(self, ch=512, wscale=0.02):
        w = chainer.initializers.Normal(wscale)
        super(Discriminator2, self).__init__()
        with self.init_scope():
            self.conv0 = L.Convolution2D(None, ch // 16, 3, 1, 1, initialW=w)
            self.conv1 = L.Convolution2D(None, ch // 8, 3, 2, 1, initialW=w)
            self.conv2 = L.Convolution2D(None, ch // 4, 3, 2, 1, initialW=w)
            self.conv3 = L.Convolution2D(None, ch // 2, 3, 2, 1, initialW=w)
            self.conv4 = L.Convolution2D(None, ch // 1, 3, 2, 1, initialW=w)

            self.fc = L.Linear(None, 1, initialW=w)

            self.bn1 = L.BatchNormalization(ch // 8, use_gamma=False)
            self.bn2 = L.BatchNormalization(ch // 4, use_gamma=False)
            self.bn3 = L.BatchNormalization(ch // 2, use_gamma=False)
            self.bn4 = L.BatchNormalization(ch // 1, use_gamma=False)

    def __call__(self, x):
        h = F.leaky_relu(add_noise(self.conv0(x)))
        h = F.leaky_relu(add_noise(self.bn1(self.conv1(h))))
        h = F.leaky_relu(add_noise(self.bn2(self.conv2(h))))
        h = F.leaky_relu(add_noise(self.bn3(self.conv3(h))))
        h = F.leaky_relu(add_noise(self.bn4(self.conv4(h))))
        return self.fc(h)


class Discriminator3(chainer.Chain):

    def __init__(self, bottom_width=4, ch=512, wscale=0.02):
        w = chainer.initializers.Normal(wscale)
        super(Discriminator3, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, ch // 8, 4, 2, 1, initialW=w)
            self.conv2 = L.Convolution2D(None, ch // 4, 4, 2, 1, initialW=w)
            self.conv3 = L.Convolution2D(None, ch // 2, 4, 2, 1, initialW=w)
            self.conv4 = L.Convolution2D(None, ch // 1, 4, 2, 1, initialW=w)

            self.fc = L.Linear(None, 1, initialW=w)

            #self.bn1 = L.BatchNormalization(ch // 8, use_gamma=False)
            self.bn2 = L.BatchNormalization(ch // 4, use_gamma=False)
            self.bn3 = L.BatchNormalization(ch // 2, use_gamma=False)
            self.bn4 = L.BatchNormalization(ch // 1, use_gamma=False)

    def __call__(self, x):
        #h = add_noise(x)
        h = F.leaky_relu(add_noise(self.conv1(x)))
        h = F.leaky_relu(add_noise(self.bn2(self.conv2(h))))
        h = F.leaky_relu(add_noise(self.bn3(self.conv3(h))))
        h = F.leaky_relu(add_noise(self.bn4(self.conv4(h))))
        return self.fc(h)

