# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils


class Generator(nn.Module):
    def __init__(self, ngpu, nz, nch_gen, nch_img):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, nch_gen, 4, 1, 0, bias=True),
            nn.BatchNorm2d(nch_gen),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nch_gen) x 4 x 4
            nn.ConvTranspose2d(nch_gen, nch_gen // 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(nch_gen // 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nch_gen/2) x 8 x 8
            nn.ConvTranspose2d(nch_gen // 2, nch_gen // 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(nch_gen // 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nch_gen/4) x 16 x 16
            nn.ConvTranspose2d(nch_gen // 4, nch_gen // 8, 4, 2, 1, bias=True),
            nn.BatchNorm2d(nch_gen // 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nch_gen/8) x 32 x 32
            nn.ConvTranspose2d(nch_gen // 8, nch_img, 4, 2, 1, bias=True),
            nn.Tanh()
            # state size. (nch_img) x 64 x 64
        )
        print(self)

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output
        

class Discriminator(nn.Module):
    def __init__(self, ngpu, nch_dis, nch_img):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nch_img) x 64 x 64
            nn.Conv2d(nch_img, nch_dis // 16, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nch_dis/16) x 64 x 64
            nn.Conv2d(nch_dis // 16, nch_dis // 8, 4, 2, 1, bias=True),
            nn.BatchNorm2d(nch_dis // 8),
            nn.Dropout2d(),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nch_dis/8) x 32 x 32
            nn.Conv2d(nch_dis // 8, nch_dis // 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(nch_dis // 4),
            nn.Dropout2d(),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nch_dis/4) x 16 x 16
            nn.Conv2d(nch_dis // 4, nch_dis // 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(nch_dis // 2),
            nn.Dropout2d(),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nch_dis/2) x 8 x 8
            nn.Conv2d(nch_dis // 2, nch_dis, 4, 2, 1, bias=True),
            nn.BatchNorm2d(nch_dis),
            nn.Dropout2d(),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nch_dis) x 4 x 4
            nn.Conv2d(nch_dis, 1, 4, 1, 0, bias=True),
            # nn.Sigmoid()
        )
        print(self)

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)

