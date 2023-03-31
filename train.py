# Copyright (c) 2023-present, Royal Bank of Canada.
# Copyright (c) 2021, Layne
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################################
# Code is based on the SAT (https://arxiv.org/abs/2201.12740) implementation
# from https://github.com/LayneH/SAT-selective-cls by Tung Nguyen 
####################################################################################


from __future__ import print_function

import argparse
import os
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
import models.cifar as models
import models.non_cifar as non_cifar_models

from tqdm import tqdm

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig, closefig
import dataset_utils, large_dataset_utils
from loss import SelfAdativeTraining, deep_gambler_loss

model_names = ("vgg16","vgg16_bn","resnet34", "EfficientNet", "resnext50_32x4d", "regnet_x_400mf", "regnet_x_800mf", "regnet_x_1_6gf", "shufflenet_v2_x1_0", "shufflenet_v2_x1_5")

parser = argparse.ArgumentParser(description='Selective Classification for Self-Adaptive Training')
parser.add_argument('-d', '--dataset', default='cifar10', type=str, choices=['cifar10', 'imagenet100', 'imagenet_subset', 'imagenet', 'cars', 'food'])
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--mode', default='train', type=str, choices=['train', 'tuning'],
                    help='mode: tuning refers to 80/20 split of the training data for hyperparameter tuning')
# Training
parser.add_argument('-t', '--train', dest='evaluate', action='store_true',
                    help='train the model. When evaluate is true, training is ignored and trained models are loaded.')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--save_model_step', default=25, type=int, metavar='N',
                    help='number of epochs to run before a model is saved')
parser.add_argument('--train-batch', default=64, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=200, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--num_classes', default=150, type=int, metavar='N',
                    help='Number of Classes for ImageNetSubset ONLY')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--schedule', type=int, nargs='+', default=[25,50,75,100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,475,500],
                        help='Multiply learning rate by gamma at the scheduled epochs (default: 25,50,75,100,125,150,175,200,225,250,275)')
parser.add_argument('--gamma', type=float, default=0.5, help='LR is multiplied by gamma on schedule (default: 0.5)') 
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--sat-momentum', default=0.9, type=float, help='momentum for sat')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('-o', '--rewards', dest='rewards', type=float, nargs='+', default=[2.2],
                    metavar='o', help='The reward o for a correct prediction; Abstention has a reward of 1. Provided parameters would be stored as a list for multiple runs.')
parser.add_argument('--pretrain', type=int, default=0,
                    help='Number of pretraining epochs using the cross entropy loss, so that the learning can always start. Note that it defaults to 100 if dataset==cifar10 and reward<6.1, and the results in the paper are reproduced.')
parser.add_argument('--coverage', type=float, nargs='+',default=[100.,99.,98.,97.,95.,90.,85.,80.,75.,70.,60.,50.,40.,30.,20.,10.],
                    help='the expected coverages used to evaluated the accuracies after abstention')                    
# Save
parser.add_argument('-s', '--save', default='save', type=str, metavar='PATH',
                    help='path to save checkpoint (default: save)')
parser.add_argument('--loss', default='sat', type=str,
                    help='loss function (sat, ce, gambler, sat_entropy)')
parser.add_argument('--entropy', type=float, default=0.0, help='Entropy Coefficient for the SAT Loss (default: 0.0)') 
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg16_bn',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: vgg16_bn) Please edit the code to train with other architectures')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate trained models on validation set, following the paths defined by "save", "arch" and "rewards"')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# set the abstention definitions
expected_coverage = args.coverage
reward_list = args.rewards

# Use CUDA
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

num_classes=10 # this is modified later in main() when defining the specific datasets

def main():
    print(args)

    # make path for the current archtecture & reward
    if not resume_path and not os.path.isdir(save_path):
        mkdir_p(save_path)

    # Dataset
    print('==> Preparing dataset %s' % args.dataset)
    global num_classes
    if args.dataset == 'cifar10':
        dataset = dataset_utils.C10
        num_classes = 10
        input_size = 32
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        trainset = dataset(root='~/datasets/CIFAR10', train=True, download=True, transform=transform_train)
        testset = dataset(root='~/datasets/CIFAR10', train=False, download=True, transform=transform_test)
    elif args.dataset == 'imagenet100':
        num_classes = 100
        input_size = 224

        train_trsfm = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        test_trsfm = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        train_data_dir = '/shared-data/imagenet/raw/train'
        test_data_dir = '/shared-data/imagenet/raw/val'

        if args.mode == 'tuning':
            print("HYPERPARAMETER TUNING MODE")
            trainset = large_dataset_utils.ImageNet100_Dataset(train_data_dir, transform=train_trsfm, split='train')
            testset = large_dataset_utils.ImageNet100_Dataset(train_data_dir, transform=test_trsfm, split='test') # Different split of train data for hyperparameter tuning
        else:
            print("Normal Training Mode")
            trainset = large_dataset_utils.ImageNet100_Dataset(train_data_dir, transform=train_trsfm)
            testset = large_dataset_utils.ImageNet100_Dataset(test_data_dir, transform=test_trsfm)
    elif args.dataset == 'imagenet_subset':
        num_classes = args.num_classes
        input_size = 224

        cur_file_path = os.path.dirname(os.path.abspath(__file__))
        # The class subset is taken from: https://github.com/HobbitLong/CMC/blob/master/imagenet100.txt
        with open(os.path.join(cur_file_path, 'imagenet_subsets', f'{num_classes}.txt')) as f:  

            class_names = list(map(lambda x : x.strip(), f.readlines()))

        train_trsfm = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        test_trsfm = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        train_data_dir = '/shared-data/imagenet/raw/train'
        test_data_dir = '/shared-data/imagenet/raw/val'
        trainset = large_dataset_utils.ImageNetSubset_Dataset(train_data_dir, class_names, transform=train_trsfm)
        testset = large_dataset_utils.ImageNetSubset_Dataset(test_data_dir, class_names, transform=test_trsfm)
    elif args.dataset == 'imagenet':
        num_classes = 1000
        input_size = 224

        train_trsfm = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        test_trsfm = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


        
        train_data_dir = '/shared-data/imagenet/raw/train'
        test_data_dir = '/shared-data/imagenet/raw/val'

        if args.mode == 'tuning':
            print("HYPERPARAMETER TUNING MODE")
            trainset = large_dataset_utils.ImageNet_Dataset(train_data_dir, transform=train_trsfm, split='train')
            testset = large_dataset_utils.ImageNet_Dataset(train_data_dir, transform=test_trsfm, split='test') # Different split of train data for hyperparameter tuning
        else:
            print("Normal Training Mode")
            trainset = large_dataset_utils.ImageNet_Dataset(train_data_dir, transform=train_trsfm)
            testset = large_dataset_utils.ImageNet_Dataset(test_data_dir, transform=test_trsfm)
    elif args.dataset == 'cars':
        num_classes = 196
        input_size = 224
        
        train_trsfm = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(35),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
            transforms.RandomGrayscale(p=0.5),
            transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
            transforms.RandomPosterize(bits=2, p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
                )
        ])
        test_trsfm = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        trainset = large_dataset_utils.Cars(root='~/datasets/cars', train=True, download=True, transform=train_trsfm)
        testset = large_dataset_utils.Cars(root='~/datasets/cars', train=False, download=True, transform=test_trsfm)
    elif args.dataset == 'food':
        num_classes = 101
        input_size = 224

        train_trsfm = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(35),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
            transforms.RandomGrayscale(p=0.5),
            transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
            transforms.RandomPosterize(bits=2, p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
                )
        ])
        test_trsfm = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        trainset = large_dataset_utils.Food(root='~/datasets/food', train=True, download=True, transform=train_trsfm)
        testset = large_dataset_utils.Food(root='~/datasets/food', train=False, download=True, transform=test_trsfm)

    # DataLoaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    
    
    # Model
    print("==> creating model '{}'".format(args.arch))

    if "cifar" not in args.dataset:
        model = non_cifar_models.__dict__[args.arch](num_classes=num_classes if args.loss == 'ce' else num_classes+1)
    else:
        model = models.__dict__[args.arch](num_classes=num_classes if args.loss == 'ce' else num_classes+1, input_size=input_size)

    if use_cuda: model = torch.nn.DataParallel(model.cuda())
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    if args.pretrain: criterion = nn.CrossEntropyLoss()
    if args.loss == 'ce':
        criterion = nn.CrossEntropyLoss() 
    elif args.loss == 'gambler':
        criterion = deep_gambler_loss
    elif args.loss == 'sat' or args.loss == 'sat_entropy':
        criterion = SelfAdativeTraining(num_examples=len(trainset), num_classes=num_classes, mom=args.sat_momentum)
    # the conventional loss is replaced by the gambler's loss in train() and test() explicitly except for pretraining
    optimizer = optim.SGD(model.parameters(), lr=state['lr'], momentum=args.momentum, weight_decay=args.weight_decay)


    title = args.dataset + '-' + args.arch + ' o={:.2f}'.format(reward)
    logger = Logger(os.path.join(save_path, 'eval.txt' if args.evaluate else 'log.txt'), title=title)
    logger.set_names(['Epoch', 'Learning Rate', 'Train Loss', 'Test Loss', 'Train Err.', 'Test Err.'])

    # if only for evaluation, the training part will not be executed
    if args.evaluate:
        print('\nEvaluation only')
        assert os.path.isfile(resume_path), 'no model exists at "{}"'.format(resume_path)
        model = torch.load(resume_path)
        if use_cuda: model = model.cuda()
        test(testloader, model, criterion, args.epochs, use_cuda, evaluation=True)
        return

    # train
    for epoch in range(0, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        print('\n'+save_path)
        print('Epoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, use_cuda)
        test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda)
        print(train_acc, test_acc)


        if (epoch+1) % args.save_model_step == 0:
            # save the model
            filepath = os.path.join(save_path, "{:d}".format(epoch+1) + ".pth")
            torch.save(model, filepath)
        
        # append logger file
        logger.append([epoch+1, state['lr'], train_loss, test_loss, 100-train_acc, 100-test_acc])

    # save the model
    filepath = os.path.join(save_path, "{:d}".format(args.epochs) + ".pth")
    torch.save(model, filepath)
    last_path = os.path.join(save_path, "{:d}".format(args.epochs-1) + ".pth")
    if os.path.isfile(last_path): os.remove(last_path)
    logger.plot(['Train Loss', 'Test Loss'])
    savefig(os.path.join(save_path, 'logLoss.eps'))
    closefig()
    logger.plot(['Train Err.', 'Test Err.'])
    savefig(os.path.join(save_path, 'logErr.eps'))
    closefig()
    logger.close()

def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    print("TrainLoader Length:", len(trainloader))
    for batch_idx,  batch_data in tqdm(enumerate(trainloader)):
        inputs, targets, indices = batch_data
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        # compute output
        outputs = model(inputs)
        if epoch >= args.pretrain:
            if args.loss == 'gambler':
                loss = criterion(outputs, targets, reward)
            elif args.loss == 'sat':
                loss = criterion(outputs, targets, indices)
            elif args.loss == 'sat_entropy':
                softmax = nn.Softmax(-1)
                loss = criterion(outputs, targets, indices) + (args.entropy * (-softmax(outputs[:, :-1]) * outputs[:, :-1]).sum(-1)).mean()
            else:
                loss = criterion(outputs, targets)
        else:
            if args.loss == 'ce':
                loss = F.cross_entropy(outputs, targets)
            else:
                loss = F.cross_entropy(outputs[:, :-1], targets)

        # measure accuracy and record loss
        if args.dataset != 'catsdogs':
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
        else:
            prec1 = accuracy(outputs.data, targets.data, topk=(1,))[0]
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def test(testloader, model, criterion, epoch, use_cuda, evaluation = False):
    global best_acc

    # whether to evaluate uncertainty, or confidence
    if evaluation:
        evaluate(testloader, model, use_cuda)
        return

    # switch to test mode
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    abstention_results = []
    sr_results = []
    for batch_idx, batch_data in enumerate(testloader):
        inputs, targets, indices = batch_data
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs = inputs.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        with torch.no_grad():
            output_logits = model(inputs).cpu()
            outputs = output_logits
            values, predictions = outputs.data.max(1)
            if epoch >= args.pretrain:
                # calculate loss
                if args.loss == 'gambler':
                    loss = criterion(outputs, targets, reward)
                elif args.loss == 'sat' or args.loss == 'sat_entropy':
                    loss = F.cross_entropy(outputs[:, :-1], targets)
                else:
                    loss = criterion(outputs, targets)
                outputs = F.softmax(outputs, dim=1)
                if args.loss == 'ce':
                    outputs, reservation = outputs, (outputs * torch.log(outputs)).sum(-1) # Reservation is neg. entropy here. 
                else:
                    outputs, reservation = outputs[:,:-1], outputs[:,-1]
                # analyze the accuracy  different abstention level
                abstention_results.extend(zip(list( reservation.numpy() ),list( predictions.eq(targets.data).numpy() )))

                pred_logits = nn.functional.softmax(output_logits[:,:-1], -1)
                sr_results.extend(zip(list(pred_logits.max(-1)[0].numpy()), list( predictions.eq(targets.data).numpy() )))
            else:
                if args.loss == 'ce':
                    loss = F.cross_entropy(outputs.cpu(), targets)
                else:
                    loss = F.cross_entropy(outputs[:,:-1].cpu(), targets)

            # measure accuracy and record loss
            if args.dataset != 'catsdogs':
                prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))
            else:
                prec1 = accuracy(outputs.data, targets.data, topk=(1,))[0]
                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    if epoch >= args.pretrain:
    	# sort the abstention results according to their reservations, from high to low
        abstention_results.sort(key = lambda x: x[0])
        # get the "correct or not" list for the sorted results
        sorted_correct = list(map(lambda x: int(x[1]), abstention_results))
        size = len(sorted_correct)
        print('Abstention Logit: accuracy of coverage ',end='')
        for coverage in expected_coverage:
            covered_correct = sorted_correct[:round(size/100*coverage)]
            print('{:.0f}: {:.3f}, '.format(coverage, sum(covered_correct)/len(covered_correct)*100.), end='')
        print('')

    	# sort the abstention results according to Softmax Response scores, from high to low
        sr_results.sort(key = lambda x: -x[0])
        # get the "correct or not" list for the sorted results
        sorted_correct = list(map(lambda x: int(x[1]), sr_results))
        size = len(sorted_correct)
        print('Softmax Response: accuracy of coverage ',end='')
        for coverage in expected_coverage:
            covered_correct = sorted_correct[:round(size/100*coverage)]
            print('{:.0f}: {:.3f}, '.format(coverage, sum(covered_correct)/len(covered_correct)*100.), end='')
        print('')
    return (losses.avg, top1.avg)

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']
            
# this function is used to evaluate the accuracy on test set per coverage
def evaluate(testloader, model, use_cuda):
    model.eval()
    abortion_results = [[],[]]
    sr_results = [[],[]]
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(testloader):
            inputs, targets = batch_data[:2]
            if use_cuda:
                inputs, targets = inputs.cuda(), targets
            output_logits = model(inputs)
            output = F.softmax(output_logits,dim=1)
            if args.loss == 'ce':
                reservation = 1 - output.data.max(1)[0].cpu()
            else:
                output, reservation = output[:,:-1], (output[:,-1]).cpu()
            values, predictions = output.data.max(1)
            predictions = predictions.cpu()
            abortion_results[0].extend(list( reservation ))
            abortion_results[1].extend(list( predictions.eq(targets.data) ))

            pred_logits = nn.functional.softmax(output_logits[:,:-1], -1)
            sr_results[0].extend(list( -pred_logits.max(-1)[0]))
            sr_results[1].extend(list( predictions.eq(targets.data) ))
    abortion_scores, abortion_correct = torch.stack(abortion_results[0]), torch.stack(abortion_results[1])
    sr_scores, sr_correct = torch.stack(abortion_results[0]), torch.stack(abortion_results[1])
    
    # Abstention Logit Results
    abortion_results = []
    bisection_method(abortion_scores, abortion_correct, abortion_results)

    print("\nAbstention\tLogit\tTest\tCoverage\tError")
    for idx, _ in enumerate(abortion_results):
        print('{:.0f},\t{:.2f},\t\t{:.3f}'.format(expected_coverage[idx], abortion_results[idx][0]*100., (1 - abortion_results[idx][1])*100))

    # Softmax Response Results
    sr_results = []
    bisection_method(sr_scores, sr_correct, sr_results)

    print("\Softmax\tResponse\tTest\tCoverage\tError")
    for idx, _ in enumerate(sr_results):
        print('{:.0f},\t{:.2f},\t\t{:.3f}'.format(expected_coverage[idx], abortion_results[idx][0]*100., (1 - abortion_results[idx][1])*100))

    return


def bisection_method(score, correct, results): 

    def calc_threshold(val_tensor,cov): # Coverage is a perentage in this input
        threshold=np.percentile(np.array(val_tensor), 100-cov*100)
        return threshold

    neg_score = -score
    for coverage in expected_coverage: # Coverage is a number from 0 to 100 here
        threshold = calc_threshold(neg_score, coverage/100)

        mask = (neg_score >= threshold)

        nData = len(correct)
        nSelected = mask.long().sum().item()
        isCorrect = correct[mask]
        nCorrectSelected = isCorrect.long().sum().item()
        passed_acc = nCorrectSelected/nSelected
        results.append((nSelected/nData, passed_acc))


if __name__ == '__main__':
    if args.loss == 'sat_entropy':
        if args.mode == 'tuning':
            base_path = os.path.join(args.save, args.dataset, args.loss, args.mode, f'entropy_coeff-{str(args.entropy)}', args.arch)
        else:
            base_path = os.path.join(args.save, args.dataset, args.loss, f'entropy_coeff-{str(args.entropy)}', args.arch)
    else:
        base_path = os.path.join(args.save, args.dataset, args.loss, args.arch)

        
    baseLR = state['lr']
    base_pretrain = args.pretrain
    resume_path = ""
    for i in range(len(reward_list)): 
        state['lr'] = baseLR
        reward = reward_list[i]
        if "imagenet_subset" == args.dataset:
            base_path = os.path.join(base_path, f"nClasses-{args.num_classes}")

        save_path = os.path.join(base_path, 'o{:.2f}'.format(reward), f"seed-{args.manualSeed}")

        if args.evaluate:
            resume_path= os.path.join(save_path,'{:d}.pth'.format(args.epochs))
        args.pretrain = base_pretrain
        
        # default the pretraining epochs to 100 to reproduce the results in the paper
        if args.loss == 'gambler' and args.pretrain == 0:
            if  args.dataset == 'cifar10' and reward < 6.3:
                args.pretrain = 100
            elif args.dataset == 'svhn' and reward < 6.0:
                args.pretrain = 50
            elif args.dataset == 'catsdogs':
                args.pretrain = 50
        
        main()
