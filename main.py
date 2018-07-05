import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
#import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import scipy
import numpy as np
from numpy.fft import fftshift
from scipy.fftpack import fft2
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')

best_prec1 = 0



def main():
    print('starting...')
    matplotlib.get_backend()
    global args, best_prec1
    args = parser.parse_args()

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    """if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()"""
    model=Net1(nombre_filtre)

    """if not args.distributed:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)"""
    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    ################criterion = nn.CrossEntropyLoss().cuda()


    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    """val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,plt.switch_backend('agg')

        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)"""

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, optimizer, epoch)
        #train(model, optimizer, epoch)
        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        print('epoch =', epoch + 1)
        print('loss = ', losses.avg)
        print('ratio regular/realloss =', ratio.avg.item())
        losses.reset()
        ratio.reset()
        # save filter images
        for k in range(0, nombre_filtre):

            for channel_ in range(0, channel):
                fig = plt.figure(1 + k * channel + channel_)
                freal = model.conv_real.weight.data[k, channel_]
                fimag = model.conv_imag.weight.data[k, channel_]
                plt.subplot(1, 2, 1)
                plt.imshow(freal)
                plt.subplot(1, 2, 2)
                plt.imshow(fimag)

                fig.savefig('/home/lucass/optimimagenet/images/filter{k}{channel_.pdf'.format(k=k, channel_=channel_))


        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)

def norme(x):               #calcule norme(x), renvoi un vecteur de longueur taille du batch
    x = x ** 2
    x = torch.sum(x, 1)
    x = torch.mean(x,1)
    x = torch.mean(x,1)
    x = torch.sqrt(x)
    return x

def myloss(input,net):
    y,p= net(input)
    loss = norme(input)-norme(y)
    loss = torch.mean(loss)          #on moyenne sur le batch
    regular_ = torch.mean(F.relu(norme(p)-norme(input)))
    ratio=regular_/loss
    loss = loss+regular_*lambda_regular
    return loss,y,p,ratio


def train(train_loader, model, optimizer, epoch):
#def train(model, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()


    losses = AverageMeter()

    ratio = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    #model.train()

    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        #target = target.cuda(non_blocking=True)

        input=input.cuda()
        # compute output

        #output = model(compress)
        loss,output,p,ratio_ = myloss(input,model)

        # measure accuracy and record loss
        ###prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        ratio.update(ratio_,input.size(0))
        ###top1.update(prec1[0], input.size(0))
        ###top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('batch :', i)
            print('Epoch: [{0}][{1}/{2}]\t',
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))#, top1=top1, top5=top5))
            print('ratio :', ratio.avg)

                  #'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  #'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'
            for k in range(0, nombre_filtre):

                for channel_ in range(0, channel):
                    fig = plt.figure(1 + k * channel + channel_)
                    freal = Mymodel.conv_real.weight.data[k, channel_]
                    fimag = Mymodel.conv_imag.weight.data[k, channel_]
                    plt.subplot(1, 2, 1)
                    plt.imshow(freal)
                    plt.subplot(1, 2, 2)
                    plt.imshow(fimag)

                    fig.savefig('/home/lucass/optimimagenet/images/filter{k}{channel_}).pdf'.format(k=k,channel_=channel_))
                                                                                                                #'channel_': channel_})


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)


            # compute output

            # output = model(compress)
            loss, output, p, ratio = myloss(input, Mymodel)

            # measure accuracy and record loss
            #prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            #top1.update(prec1[0], input.size(0))
            #top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      )
                      #'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      #'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       #i, len(val_loader), batch_time=batch_time, loss=losses,
                       #top1=top1, top5=top5))

        #print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
         #     .format(top1=top1, top5=top5))

    return #top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

lum_seul=False
if lum_seul:
    channel=1
else: channel=3
nombre_filtre=2
padding_=5
kernel=2*padding_+1
learning_rate=0.005
lambda_regular=1

def gabor_2d(M, N, sigma, theta, xi, slant=1.0, offset=0, fft_shift=None):
    gab = np.zeros((M, N), np.complex64)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], np.float32)
    R_inv = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]], np.float32)
    D = np.array([[1, 0], [0, slant * slant]])
    curv = np.dot(R, np.dot(D, R_inv)) / ( 2 * sigma * sigma)

    for ex in [-2, -1, 0, 1, 2]:
        for ey in [-2, -1, 0, 1, 2]:
            [xx, yy] = np.mgrid[offset + ex * M:offset + M + ex * M, offset + ey * N:offset + N + ey * N]
            arg = -(curv[0, 0] * np.multiply(xx, xx) + (curv[0, 1] + curv[1, 0]) * np.multiply(xx, yy) + curv[
                1, 1] * np.multiply(yy, yy)) + 1.j * (xx * xi * np.cos(theta) + yy * xi * np.sin(theta))
            gab = gab + np.exp(arg)

    norm_factor = (2 * 3.1415 * sigma * sigma / slant)
    gab = gab / norm_factor

    if (fft_shift):
        gab = np.fft.fftshift(gab, axes=(0, 1))
    return gab

phi=gabor_2d(kernel,kernel,1,0,0,fft_shift=True)
phi=np.absolute(phi)
phi_chap=fft2(phi)
phi_chap=np.absolute(phi)
phi=torch.from_numpy(phi)
#print(torch.sum(phi))

avg=nn.Conv2d(nombre_filtre,nombre_filtre,kernel,bias=False,padding=padding_,groups=nombre_filtre)


for k1 in range(0,nombre_filtre):
    avg.weight.data[k1,0,:,:]=phi

avg = torch.nn.DataParallel(avg).cuda()

class Net1(nn.Module):
    def __init__(self, nombre_filtre):
        super(Net1, self).__init__()
        self.conv_real = nn.Conv2d(channel, nombre_filtre, kernel, bias=False,padding=padding_)
        self.conv_imag = nn.Conv2d(channel, nombre_filtre, kernel, bias=False, padding=padding_)

        #self.avg = torch.nn.AvgPool2d(5,stride=1,padding=2,count_include_pad=True)

    def forward(self, x):
        y_r = self.conv_real(x)
        y_i = self.conv_imag(x)

        y_r = y_r ** 2
        y_i = y_i ** 2
        y = y_r + y_i
        y = torch.sqrt(y)
        p = y                       #p stock le module au carre de Wx
        y = avg(y)

        return y,p



if __name__ == '__main__':
    main()