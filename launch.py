import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset

import sys
import models
import datasets
import trainer
import logging
from torchinfo import summary
from datetime import timedelta
from datetime import datetime
import json
from utils.utils import AverageMeter, ProgressMeter, Summary
from torch.utils.tensorboard import SummaryWriter


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
def prepare_ddp_args(args):
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    return args

def prepare_seed_env(args):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    return args

def _backup_files(source_dir, backup_dir, exclude_dirs=['exp', 'temp', 'output']):
    if exclude_dirs is None:
        exclude_dirs = []

    # 遍历当前目录及其子目录
    for root, dirs, files in os.walk(source_dir):
        # 检查当前目录是否在排除列表中
        if any(excluded in root for excluded in exclude_dirs):
            continue
        
        for file in files:
            if file.endswith('.py'):
                # 构建源文件的完整路径
                source_file_path = os.path.join(root, file)
                
                # 构建目标文件的完整路径
                relative_path = os.path.relpath(root, source_dir)
                target_dir = os.path.join(backup_dir, relative_path)
                target_file_path = os.path.join(target_dir, file)
                
                # 创建目标目录（如果不存在）
                os.makedirs(target_dir, exist_ok=True)
                
                # 复制文件
                shutil.copy2(source_file_path, target_file_path)
            
def prepare_path_env(args):
    # 获取模型名称和数据集名称
    model_name = args.model
    dataset_name = args.dataset_name
    prefix = args.prefix
    extra_note = args.extra_note

    # 获取当前时间戳
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # 组合文件夹路径
    if prefix != "":
        output_dir = os.path.join("exp", f"{prefix}_{dataset_name}_{model_name}_{timestamp}_{extra_note}")
    else:
        output_dir = os.path.join("exp", f"{dataset_name}_{model_name}_{timestamp}_{extra_note}")

    # 创建文件夹
    os.makedirs(output_dir, exist_ok=True)

    # 备份文件
    backup_dir = os.path.join(output_dir, "backup")
    os.makedirs(backup_dir, exist_ok=True)

    # if os.path.dirname(os.path.dirname(backup_dir)) != os.getcwd():
    #     raise ValueError
    _backup_files(os.getcwd(), backup_dir)

    # 备份参数
    with open(os.path.join(output_dir, 'args.json'), 'w') as json_file:
        json.dump(vars(args), json_file, indent=4)
        
    return output_dir

def build_model(args):
    model = models.__dict__[args.model]()
    # model_info = summary(model, input_size=(1, 3, 224, 224))
    model_info = summary(model)
    return model, model_info

def build_dataset(args):
    # 构建数据集
    train_dataset, val_dataset, collate_fn = datasets.__dict__[args.dataset_name](args)

    # 准备训练集信息
    train_info = "Training Dataset:\n"
    train_info += f" - Number of samples: {len(train_dataset)}\n"
    sample = train_dataset[0]
    if isinstance(sample, tuple):
        train_info += f" - Sample shape (input): {sample[0].shape}\n"
        train_info += f" - Sample shape (target): {sample[1].shape if isinstance(sample[1], torch.Tensor) else 'Not a tensor'}\n"
    else:
        train_info += f" - Sample shape: {sample.shape}\n"

    # 准备验证集信息
    val_info = "Validation Dataset:\n"
    val_info += f" - Number of samples: {len(val_dataset)}\n"
    sample = val_dataset[0]
    if isinstance(sample, tuple):
        val_info += f" - Sample shape (input): {sample[0].shape}\n"
        val_info += f" - Sample shape (target): {sample[1].shape if isinstance(sample[1], torch.Tensor) else 'Not a tensor'}\n"
    else:
        val_info += f" - Sample shape: {sample.shape}\n"

    # 将信息合并成一个字符串
    dataset_info = train_info + "\n" + val_info

    return train_dataset, val_dataset, collate_fn, dataset_info

def build_dataloader(args, train_dataset, val_dataset, collate_fn=None):
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, collate_fn=collate_fn)

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, collate_fn=collate_fn)
    
    return train_dataloader, val_dataloader

def build_optimizer(model, args):
    optimizer_name = args.optimizer.lower()
    learning_rate = args.lr
    weight_decay = args.weight_decay if hasattr(args, 'weight_decay') else 0.0
    momentum = args.momentum if hasattr(args, 'momentum') else 0.9

    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    optimizer_info = (
        f"Optimizer: {optimizer_name.upper()}\n"
        f" - Learning Rate: {learning_rate}\n"
        f" - Weight Decay: {weight_decay}\n"
    )

    if optimizer_name in ['sgd', 'rmsprop']:
        optimizer_info += f" - Momentum: {momentum}\n"

    return optimizer, optimizer_info

def build_scheduler(optimizer, args):
    # 从args获取调度器类型和其他参数
    scheduler_name = args.scheduler.lower()
    step_size = args.step_size if hasattr(args, 'step_size') else 30
    gamma = args.gamma if hasattr(args, 'gamma') else 0.1
    milestones = args.milestones if hasattr(args, 'milestones') else [30, 60, 90]
    T_max = args.T_max if hasattr(args, 'T_max') else 50
    eta_min = args.eta_min if hasattr(args, 'eta_min') else 0

    # 根据调度器名称创建相应的学习率调度器
    if scheduler_name == 'steplr':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        scheduler_info = (
            f"Scheduler: StepLR\n"
            f" - Step Size: {step_size}\n"
            f" - Gamma: {gamma}\n"
        )
    elif scheduler_name == 'multisteplr':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        scheduler_info = (
            f"Scheduler: MultiStepLR\n"
            f" - Milestones: {milestones}\n"
            f" - Gamma: {gamma}\n"
        )
    elif scheduler_name == 'exponentiallr':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        scheduler_info = (
            f"Scheduler: ExponentialLR\n"
            f" - Gamma: {gamma}\n"
        )
    elif scheduler_name == 'cosineannealinglr':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        scheduler_info = (
            f"Scheduler: CosineAnnealingLR\n"
            f" - T_max: {T_max}\n"
            f" - Eta Min: {eta_min}\n"
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    return scheduler, scheduler_info

def build_criterion(args, device):

    # 根据args中的设置选择损失函数
    if args.criterion == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    elif args.criterion == 'mse':
        criterion = nn.MSELoss()
    elif args.criterion == 'l1':
        criterion = nn.L1Loss()
    elif args.criterion == 'bce':
        criterion = nn.BCELoss()
    else:
        raise ValueError(f"unspported loss: {args.criterion}")

    # 将损失函数移动到指定设备上
    criterion = criterion.to(device)

    return criterion

def bulid_logger(base_dir):
    log_file_path = os.path.join(base_dir, "log.txt")

    # 创建一个 logger 对象
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # 设置最低日志级别

    # 创建一个文件处理器，用于将日志写入文件
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)

    # 创建一个控制台处理器，用于将日志输出到控制台
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 将处理器添加到 logger 对象
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    tb_logger = SummaryWriter(log_dir=os.path.join(base_dir))

    return logger, tb_logger

def bulid_evaluator():
    pass

def train(args, model, train_dataloader, val_dataloader, optimizer, scheduler, criterion, save_dir, tb_logger):
    best_acc1 = 0
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train_loss, train_acc1, train_acc5 = train_epoch(args=args, 
                                                         model=model, 
                                                         train_dataloader=train_dataloader, 
                                                         optimizer=optimizer, 
                                                         criterion=criterion,
                                                         epoch=epoch,
                                                         device=device,
                                                         tb_logger=tb_logger,
                                                        )

        # evaluate on validation set
        valid_loss, valid_acc1, valid_acc5 = validate(val_dataloader, model, criterion, args)
        tb_logger.add_scalar('valid_loss', valid_loss, epoch)
        tb_logger.add_scalar('valid_acc1', valid_acc1, epoch)
        tb_logger.add_scalar('valid_acc5', valid_acc5, epoch)

        scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = valid_acc1 > best_acc1
        best_acc1 = max(valid_acc1, best_acc1)

        save_checkpoint({
                'epoch': epoch + 1,
                'model': args.model,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            }, is_best, save_dir=save_dir)

def validate(val_loader, model, criterion, args):

    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                if args.gpu is not None and torch.cuda.is_available():
                    images = images.cuda(args.gpu, non_blocking=True)
                # if torch.backends.mps.is_available():
                #     images = images.to('mps')
                #     target = target.to('mps')
                if torch.cuda.is_available():
                    target = target.cuda(args.gpu, non_blocking=True)

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        run_validate(aux_val_loader, len(val_loader))

    progress.display_summary()

    return losses.avg, top1.avg, top5.avg

def train_epoch(args, model, train_dataloader, optimizer, criterion, device, epoch, tb_logger):
    # default meter
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    # metric meter
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    progress = ProgressMeter(
        len(train_dataloader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()

    total_batches = len(train_dataloader) * args.epochs
    completed_batches = len(train_dataloader) * (epoch - 1)

    running_loss = 0.0

    for i, (images, target) in enumerate(train_dataloader):
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

       # 计算整个训练过程的ETA
        remaining_batches = total_batches - (completed_batches + i + 1)
        eta_seconds = batch_time.avg * remaining_batches
        eta_str = str(timedelta(seconds=int(eta_seconds)))

        running_loss += loss.item()
        if i % args.print_freq == 0:
            progress.display(i, eta_str)
            tb_logger.add_scalar('training loss', running_loss / args.print_freq, epoch * len(train_dataloader) + i)
            running_loss = 0.0
    
    return losses.avg, top1.avg, top5.avg
    
def save_checkpoint(state, is_best, save_dir='output', filename='checkpoint.pth.tar'):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    torch.save(state, save_path)
    if is_best:
        shutil.copyfile(save_path, os.path.join(save_dir, 'model_best.pth.tar'))

def train_step():
    pass

def val():
    pass

def val_step():
    pass

def save():
    pass

def log():
    pass

def log_setp():
    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    # ddp env
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                            'N processes per node, which has N GPUs. This is the '
                            'fastest way to use PyTorch for either single node or '
                            'multi node data parallel training')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    
    # optimizer args
    parser.add_argument('--start_epoch', default=0, type=int,)
    parser.add_argument('--optimizer', default="SGD", type=str,)
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    # parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
    #                     help='evaluate model on validation set')
    parser.add_argument('--criterion', default="cross_entropy", type=str,)
    # schduler args
    parser.add_argument('--scheduler', default="steplr", type=str,)

    # model args
    parser.add_argument('--model',default='resnet')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    
    # dataloader args
    parser.add_argument('--dataset_name',default='', help='path to dataset (default: imagenet)')
    parser.add_argument('--data_dir',default='', help='path to dataset (default: imagenet)')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')

    # log args
    parser.add_argument('--enable_wandb', action='store_true', help="enable wandb logging")
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    
    # control args
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--output_dir', type=str, default='output', help="output dir")
    parser.add_argument('--prefix', type=str, default='', help="output dir")
    parser.add_argument('--extra_note', type=str, default='default', help="")
    parser.add_argument('--evaluator', type=list, default=['acc'])



    # args = parser.parse_args()
    config = parser.parse_args()

    # # 设置ddp等参数
    # args = prepare_ddp_args(args)

    # # 设置seed等参数
    # args = prepare_seed_env(args)

    # device = torch.device('cuda:{}'.format(args.gpu))

    # exp_base_dir = prepare_path_env(args)
    
    # logger, tb_logger = bulid_logger(exp_base_dir)

    # logger.info("**************************Start building model*******************************")
    # model, model_info = build_model(args)
    # logger.info(model_info)
    # model.cuda()

    # logger.info("************************Start building dataset*************************")
    # train_dataset, val_dataset, collate_fn, dataset_info = build_dataset(args)
    # logger.info(dataset_info)

    # logger.info("************************Start building dataloader**********************")
    # train_dataloader, val_dataloader = build_dataloader(args, train_dataset, val_dataset, collate_fn=collate_fn)

    # logger.info("************************Start building optimizer***********************")
    # optimizer, optimizer_info = build_optimizer(model, args)
    # logger.info(optimizer_info)

    # logger.info("************************Start building scheduler***********************")
    # scheduler, scheduler_info = build_scheduler(optimizer, args)
    # logger.info(scheduler_info)

    # logger.info("************************Start building criterion***********************")
    # criterion = build_criterion(args, device)
    # logger.info(criterion)

    # logger.info("================================Begin to Train================================")
    # # criterion = nn.CrossEntropyLoss().to(device)
    # ckpt_save_dir = os.path.join(exp_base_dir, "ckpt")

    # train(
    #       args=args, 
    #       model=model, 
    #       train_dataloader=train_dataloader, 
    #       val_dataloader=val_dataloader, 
    #       optimizer=optimizer, 
    #       scheduler=scheduler, 
    #       criterion=criterion,
    #       save_dir=ckpt_save_dir,
    #       tb_logger=tb_logger,
    #       )

    trainer = trainer.BaseTrainer(config=config)
    trainer.train()
    
    


