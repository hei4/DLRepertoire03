# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
from scipy.misc import imresize

import chainer
from chainer import training
from chainer.training import extensions
from chainer.dataset.convert import concat_examples

#from net import Discriminator
#from net import Generator, Generator2
import net
from updater import DCGANUpdater
from visualize import out_generated_image


def main():
    parser = argparse.ArgumentParser(description='LSGAN')

    parser.add_argument('--batchsize', '-b', type=int, default=20,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=1000,
                        help='Number of sweeps over the dataset to train')
                        
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--dataset', '-i', default='',
                        help='Directory of image files.  Default is cifar-10.')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
                        
    parser.add_argument('--n_hidden', '-n', type=int, default=100,
                        help='Number of hidden units (z)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed of z at visualization stage')
    parser.add_argument('--image_size', type=int, default=64,
                        help='Size of output image')

    parser.add_argument('--display_interval', type=int, default=100,
                        help='Interval of displaying log to console (iter)')
    parser.add_argument('--preview_interval', type=int, default=1,
                        help='Interval of preview (epoch)')
    parser.add_argument('--snapshot_interval', type=int, default=10,
                        help='Interval of snapshot (epoch)')

    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# n_hidden: {}'.format(args.n_hidden))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train
    gen = net.Generator(n_hidden=args.n_hidden, image_size=args.image_size)
    # dis = Discriminator()
    dis = net.Discriminator2()
    # dis = net.Discriminator3()

    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        gen.to_gpu()  # Copy the model to the GPU
        dis.to_gpu()

    # Setup an optimizer
    def make_optimizer(model, alpha=0.0002, beta1=0.5):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001), 'hook_dec')
        return optimizer
    opt_gen = make_optimizer(gen)
    opt_dis = make_optimizer(dis)

    if args.dataset == '':
        # Load the CIFAR10 dataset if args.dataset is not specified
        train, _ = chainer.datasets.get_cifar10(withlabel=False, scale=255.)
    else:
        all_files = os.listdir(args.dataset)
        image_files = [f for f in all_files if ('png' in f or 'jpg' in f)]
        print('{} contains {} image files'.format(args.dataset, len(image_files)))
        train = chainer.datasets.ImageDataset(paths=image_files, root=args.dataset)

    # train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    train_iter = chainer.iterators.MultiprocessIterator(train, args.batchsize, n_processes=4)

    def resize_converter(batch, device=None, padding=None):
        new_batch = []
        for image in batch:
            C, W, H = image.shape
            
            if C == 4:
                image = image[:3, :, :]

            if W < H:
                offset = (H - W) // 2
                image = image[:, :, offset:offset+W]
            elif W > H:
                offset = (W - H) // 2
                image = image[:, offset:offset+H, :]
                                
            image = image.transpose(1, 2, 0)
            image = imresize(image, (args.image_size, args.image_size), interp='bilinear')
            image = image.transpose(2, 0, 1)

            image = image / 255.    # 0. ~ 1.

            # Augumentation... Random vertical flip
            if np.random.rand() < 0.5:
                image = image[:, :, ::-1]

            # Augumentation... Tone correction
            mode = np.random.randint(4)
            # mode == 0 -> no correction
            if mode == 1:
                gain = 0.2 * np.random.rand() + 0.9   # 0.9 ~ 1.1
                image = np.power(image, gain)
            elif mode == 2:
                gain = 1.5 * np.random.rand() + 1e-10     # 0 ~ 1.5
                image = np.tanh(gain * (image - 0.5))
                
                range_min = np.tanh(gain * (-0.5))      # @x=0.5
                range_max = np.tanh(gain * 0.5)         # @x=1.0
                image = (image - range_min) / (range_max - range_min)
            elif mode == 3:
                gain = 2.0 * np.random.rand() + 1e-10     # 0 ~ 1.5
                image = np.sinh(gain * (image - 0.5))
                
                range_min = np.tanh(gain * (-0.5))      # @x=0.5
                range_max = np.tanh(gain * 0.5)         # @x=1.0
                image = (image - range_min) / (range_max - range_min)

            image = 2. * image - 1.
            new_batch.append(image.astype(np.float32))
        return concat_examples(new_batch, device=device, padding=padding)

    # Set up a trainer
    updater = DCGANUpdater(
        models=(gen, dis),
        iterator=train_iter,
        optimizer={
            'gen': opt_gen,
            'dis': opt_dis
        },
        device=args.gpu,
        converter=resize_converter)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    display_interval = (args.display_interval, 'iteration')
    preview_interval = (args.preview_interval, 'epoch')
    snapshot_interval = (args.snapshot_interval, 'epoch')

    trainer.extend(
        extensions.snapshot(filename='snapshot_iter_{.updater.iteration}.npz'),
            trigger=snapshot_interval)

    trainer.extend(
        extensions.snapshot_object(gen, 'gen_iter_{.updater.iteration}.npz'),
        trigger=snapshot_interval)
    trainer.extend(
        extensions.snapshot_object(dis, 'dis_iter_{.updater.iteration}.npz'),
        trigger=snapshot_interval)

    trainer.extend(extensions.LogReport(trigger=display_interval))

    trainer.extend(
        extensions.PrintReport(['epoch', 'iteration', 'gen/loss', 'dis/loss',]),
        trigger=display_interval)

    trainer.extend(extensions.ProgressBar(update_interval=10))

    trainer.extend(
        out_generated_image(gen, dis, 10, 10, args.seed, args.out),
        trigger=preview_interval)

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
