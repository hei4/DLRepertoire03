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

from net import Generator, Discriminator


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | fake')
    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=50, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--nch_gen', type=int, default=512)
    parser.add_argument('--nch_dis', type=int, default=512)
    parser.add_argument('--nepoch', type=int, default=1000, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--gen', default='', help="path to gen (to continue training)")
    parser.add_argument('--dis', default='', help="path to dis (to continue training)")
    parser.add_argument('--outf', default='./result', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')

    args = parser.parse_args()
    print(args)

    try:
        os.makedirs(args.outf)
    except OSError:
        pass

    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", args.manualSeed)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    if args.dataset in ['imagenet', 'folder', 'lfw']:
        # folder dataset
        dataset = dset.ImageFolder(root=args.dataroot,
                                   transform=transforms.Compose([
                                       transforms.Resize(args.imageSize),
                                       transforms.CenterCrop(args.imageSize),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
    elif args.dataset == 'lsun':
        dataset = dset.LSUN(root=args.dataroot, classes=['bedroom_train'],
                            transform=transforms.Compose([
                                transforms.Resize(args.imageSize),
                                transforms.CenterCrop(args.imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    elif args.dataset == 'cifar10':
        dataset = dset.CIFAR10(root=args.dataroot, download=True,
                               transform=transforms.Compose([
                                   transforms.Resize(args.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))     # [0, +1] -> [-1, +1]
    elif args.dataset == 'fake':
        dataset = dset.FakeData(image_size=(3, args.imageSize, args.imageSize),
                                transform=transforms.ToTensor())
    
    assert dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize,
                                             shuffle=True, num_workers=int(args.workers))

    device = torch.device("cuda:0" if args.cuda else "cpu")
    nch_img = 3

    # custom weights initialization called on gen and dis
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
            m.bias.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            # m.bias.data.normal_(1.0, 0.02)
            # m.bias.data.fill_(0)

    gen = Generator(args.ngpu, args.nz, args.nch_gen, nch_img).to(device)
    gen.apply(weights_init)
    if args.gen != '':
        gen.load_state_dict(torch.load(args.gen))

    dis = Discriminator(args.ngpu, args.nch_dis, nch_img).to(device)
    dis.apply(weights_init)
    if args.dis != '':
        dis.load_state_dict(torch.load(args.dis))

    # criterion = nn.BCELoss()
    criterion = nn.MSELoss()
    
    # fixed_z = torch.randn(args.batchSize, args.nz, 1, 1, device=device)
    fixed_z = torch.randn(8*8, args.nz, 1, 1, device=device)
    a_label = 0
    b_label = 1
    c_label = 1

    # setup optimizer
    optim_dis = optim.Adam(dis.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optim_gen = optim.Adam(gen.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    
    for epoch in range(args.nepoch):
        for itr, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            dis.zero_grad()
            real_img = data[0].to(device)
            batch_size = real_img.size(0)
            label = torch.full((batch_size,), b_label, device=device)

            dis_real = dis(real_img)
            loss_dis_real = criterion(dis_real, label)
            loss_dis_real.backward()
            
            # train with fake
            z = torch.randn(batch_size, args.nz, 1, 1, device=device)
            fake_img = gen(z)
            label.fill_(a_label)
            
            dis_fake1 = dis(fake_img.detach())
            loss_dis_fake = criterion(dis_fake1, label)
            loss_dis_fake.backward()
            
            loss_dis = loss_dis_real + loss_dis_fake
            optim_dis.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            gen.zero_grad()
            label.fill_(c_label)  # fake labels are real for generator cost
            
            dis_fake2 = dis(fake_img)
            loss_gen = criterion(dis_fake2, label)
            loss_gen.backward()
            optim_gen.step()
            
            if (itr + 1) % 100 == 0:
                print('[{}/{}][{}/{}] LossD:{:.4f} LossG:{:.4f} D(x):{:.4f} D(G(z)):{:.4f}/{:.4f}'.format(
                      epoch + 1, args.nepoch, itr + 1, len(dataloader),
                      loss_dis.item(), loss_gen.item(),
                      dis_real.mean().item(), dis_fake1.mean().item(), dis_fake2.mean().item()))
            # loop end iteration

        if epoch == 0:
            vutils.save_image(real_img,
                              '{}/real_samples.png'.format(args.outf),
                              normalize=True)
        
        fake_img = gen(fixed_z)
        vutils.save_image(fake_img.detach(),
                          '{}/fake_samples_epoch_{:04}.png'.format(args.outf, epoch),
                          normalize=True)
        
        # do checkpointing
        torch.save(gen.state_dict(), '{}/gen_epoch_{}.pth'.format(args.outf, epoch))
        torch.save(dis.state_dict(), '{}/dis_epoch_{}.pth'.format(args.outf, epoch))
        # loop end epoch


if __name__ == '__main__':
    main()

