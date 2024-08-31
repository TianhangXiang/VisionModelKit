
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
import torchvision.models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset

import sys
import models
import datasets
import logging
from torchinfo import summary
from datetime import timedelta
from datetime import datetime
import json
from utils.utils import AverageMeter, ProgressMeter, Summary
from torch.utils.tensorboard import SummaryWriter
from utils.eval import *
from utils.distribute import *

class BaseTrainer():
    def __init__(self, config) -> None:
        # ddp args
        self.is_distributed = False
        if 'WORLD_SIZE' in os.environ: 
            self.is_distributed = int(os.environ['WORLD_SIZE']) > 1

        if self.is_distributed:
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

        # prepare_save_dir, setup_logger, backup_file
        self.config = config
        self.start_epoch = config.start_epoch
        self.max_epoch = config.epochs
        self._init_(config)

        self.logger.info('Initializing model...')
        self.model = self.build_model(config)
        self.logger.info('Model initialized as follow: \n{}'.format(self.model))
        num_params, num_trainable_params= self.count_param(self.model)
        self.logger.info('The #Params of model is: {:.2f}M (trainable: {:.2f}M)'.format(num_params / 1e6, num_trainable_params / 1e6))

        self.logger.info('Initializing dataset...')
        self.train_dataset, self.val_dataset, collate_fn = self.build_dataset(config)
        self.logger.info('The length of train dataset is {}'.format(len(self.train_dataset)))
        self.logger.info('The length of val dataset is {}'.format(len(self.val_dataset)))

        self.logger.info('Initializing dataloader...')
        self.train_dataloader, self.val_dataloader = self.build_dataloader(config=config, 
                                                                           train_dataset=self.train_dataset, 
                                                                           val_dataset=self.val_dataset, 
                                                                           collate_fn=collate_fn,
                                                                           )
        
        self.logger.info('The batch number of train dataloader is {}'.format(len(self.train_dataloader)))
        self.logger.info('The batch number val dataloader is {}'.format(len(self.val_dataloader)))
        
        self.logger.info('Initializing optimizer...')
        self.optimizer = self.build_optimizer(config)
        self.logger.info('The paramter of optimizer is: \n{}'.format(self.optimizer))

        self.logger.info('Initializing scheduler...')
        self.scheduler = self.build_scheduler(config, self.optimizer)
        self.logger.info('The paramter of scheduler is: \n{}'.format(self.scheduler.state_dict()))

        self.logger.info('Initializing criterion...')
        self.criterion = self.build_criterion(config)
        self.logger.info('Use criterion: {}'.format(self.criterion))

        self.logger.info('Initializing evaluator...')
        self.build_evaluator(config)
        for evaluator in self.evaluator:
            self.logger.info('Use evaluator {}'.format(evaluator))

        self.logger.info('Initializing ddp training...')
        self.handle_ddp_training()
        
    def build_model(self, config):
        model = models.__dict__[config.model]()
        if config.pretrained:
            # TODO
            pass
        # model = torchvision.models.resnet50(pretrained=True)
        # from torchinfo import summary
        # model_info = summary(model, input_size=(1, 3, 224, 224), col_names=["input_size", "output_size"])
        return model
    
    def build_dataset(self, config):
        train_dataset, val_dataset, collate_fn = datasets.__dict__[config.dataset_name](config)
        return train_dataset, val_dataset, collate_fn

    def build_dataloader(self, config, train_dataset, val_dataset, collate_fn=None):
        if self.is_distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
        else:
            train_sampler = None
            val_sampler = None

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=(train_sampler is None),
            num_workers=config.workers, pin_memory=True, sampler=train_sampler, collate_fn=collate_fn)

        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False,
            num_workers=config.workers, pin_memory=True, sampler=val_sampler, collate_fn=collate_fn)
        
        return train_dataloader, val_dataloader

    def build_optimizer(self, config):
        optimizer_name = config.optimizer.lower()
        learning_rate = config.lr
        weight_decay = config.weight_decay if hasattr(config, 'weight_decay') else 0.0
        momentum = config.momentum if hasattr(config, 'momentum') else 0.9

        if optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'rmsprop':
            optimizer = torch.optim.RMSprop(
                self.model.parameters(),
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        return optimizer
    
    def build_evaluator(self, config):
        self.evaluator = {}
        evaluator_key_list = config.evaluator
        for evaluator_key in evaluator_key_list:
            if evaluator_key.lower() == "acc":
                self.evaluator.update({
                    "acc_evaluator": Acc(),
                })

    def build_scheduler(self, config, optimizer):
        scheduler_name = config.scheduler.lower()
        step_size = config.step_size if hasattr(config, 'step_size') else 30
        gamma = config.gamma if hasattr(config, 'gamma') else 0.1
        milestones = config.milestones if hasattr(config, 'milestones') else [30, 60, 90]
        T_max = config.T_max if hasattr(config, 'T_max') else 50
        eta_min = config.eta_min if hasattr(config, 'eta_min') else 0

        if scheduler_name == 'steplr':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        elif scheduler_name == 'multisteplr':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        elif scheduler_name == 'exponentiallr':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        elif scheduler_name == 'cosineannealinglr':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")

        return scheduler
    
    def build_criterion(self, config):
        if config.criterion == 'cross_entropy':
            criterion = nn.CrossEntropyLoss()
        elif config.criterion == 'mse':
            criterion = nn.MSELoss()
        elif config.criterion == 'l1':
            criterion = nn.L1Loss()
        elif config.criterion == 'bce':
            criterion = nn.BCELoss()
        else:
            raise ValueError(f"unspported loss: {config.criterion}")
        return criterion
    
    def _init_(self, config):
        # prepare save dir
        model_name = config.model
        dataset_name = config.dataset_name
        prefix = config.prefix
        extra_note = config.extra_note

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        if prefix != "":
            output_dir = os.path.join("exp", f"{prefix}_{dataset_name}_{model_name}_{timestamp}_{extra_note}")
        else:
            output_dir = os.path.join("exp", f"{dataset_name}_{model_name}_{timestamp}_{extra_note}")

        output_dir = os.path.join(os.getcwd(), output_dir)
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

        # backup code
        if self.rank == 0:
            backup_dir = os.path.join(output_dir, "backup")
            os.makedirs(backup_dir, exist_ok=True)
            self._backup_files(os.getcwd(), backup_dir)
            # backup config 
            with open(os.path.join(output_dir, 'config.json'), 'w') as json_file:
                json.dump(vars(config), json_file, indent=4)

        # setup logger
        self.logger, self.tb_logger = self.bulid_logger(output_dir)
        self.logger.info("========> The base dir is: \n {} ".format(self.output_dir))
        self.logger.info("========> Use the follow config: \n {} ".format(config))

        # setup recoder
        self.recoder_key = config.recoder_key
        self.recoder = {
                        self.recoder_key: 0.0,
                        }
    
    def handle_ddp_training(self):
        if not self.is_distributed:
            self.model.cuda()
            device = next(self.model.parameters()).device
            self.criterion.to(device)
            self.logger.info("Use single card: {} for accelerate".format(device))
        else:
            self.model.to(self.rank)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.config.local_rank], output_device=self.config.local_rank)
            device = torch.device(f'cuda:{self.config.local_rank}')
            self.criterion = self.criterion.to(device)

    def train(self):
        for epoch in range(self.start_epoch, self.max_epoch):
            # train for one epoch
            self.train_epoch(epoch)
            self.scheduler.step()
            self.train_log_after_epoch(epoch)
            # val for one epoch
            self.val(epoch)

    def train_one_epoch(self):
        pass

    def train_step(self, batch):
        pass

    def train_log_in_epoch(self, log_dict):
        pass

    def train_log_after_epoch(self, epoch):
        if self.rank == 0:
            current_lr = self.optimizer.param_groups[0]['lr']
            self.tb_logger.add_scalar('epoch/lr', current_lr, epoch)

    @torch.no_grad()
    def val(self, epoch=-1):
        for _, evaluator in self.evaluator.items():
            evaluator.reset()
        # NOTE: self.model.eval() is IMPORTRANT, or the results will be INCORRECT!!!
        self.model.eval()

        loss_dict, metric_dict = self.val_epoch()

        self.log_after_val_epoch(loss_dict, metric_dict, epoch)

        self.save_after_val_epoch(metric_dict, epoch=epoch)
        
    def val_one_epoch():
        pass
    
    def val_log_in_epoch(self, log_dict):
        pass
    
    def val_log_after_epoch(self, log_dict):
        pass

    def train_epoch(self, epoch):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        self.model.train()

        end = time.time()

        total_batches = len(self.train_dataloader) * self.max_epoch
        completed_batches = len(self.train_dataloader) * (epoch - 1)

        device = next(self.model.parameters()).device

        if self.is_distributed:
            self.train_dataloader.sampler.set_epoch(epoch)

        for i, (images, target) in enumerate(self.train_dataloader):
            # measure data loading time
            data_time.update(time.time() - end)

            # move data to the same device as model
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output
            output = self.model(images)
            loss = self.criterion(output, target)

            batch_size = images.shape[0]
            metric_dict = {}
            for evaluator_name, evaluator in self.evaluator.items():
                evaluator.update(output, target, batch_size)
                for metric_key, metricAvgMeter in evaluator.metrics.items():
                    metric_dict.update({
                        metric_key + "_val": metricAvgMeter.val.item(),
                        metric_key + "_avg": metricAvgMeter.avg.item(),
                    })

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()

            # wait for all the gradient to be calculated
            torch.cuda.synchronize()

            self.optimizer.step()

            if self.is_distributed:
                loss = reduce_tensor(loss.data, self.world_size)

            losses.update(loss.item(), batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # compute ETA
            remaining_batches = total_batches - (completed_batches + i + 1)
            eta_seconds = batch_time.avg * remaining_batches
            eta_str = str(timedelta(seconds=int(eta_seconds)))

            if i % self.config.print_freq == 0 and ((self.is_distributed and dist.get_rank() == 0) or not self.is_distributed):
                batch_time_val_avg_str = "{:.3f} ({:.3f})".format(batch_time.val, batch_time.avg)
                data_time_val_avg_str = "{:.3f} ({:.3f})".format(data_time.val, data_time.avg)
                loss_val_avg_str = "{:.3f} ({:.3f})".format(losses.val, losses.avg)
                current_lr = self.optimizer.param_groups[0]['lr']

                time_str = "Epoch: [{} / {}][{} / {}] Batch time: {} Data time: {} ETA: {}".format(epoch, self.max_epoch, i, len(self.train_dataloader), batch_time_val_avg_str, data_time_val_avg_str, eta_str)
                loss_str = "Loss: {} LR: {:.5f}".format(loss_val_avg_str, current_lr)
                metric_str = ""
                for mertic_key, metric_value in metric_dict.items():
                    metric_str += "{}: {:.3f}  ".format(mertic_key, metric_value)
                    self.tb_logger.add_scalar(f'train/{metric_key}', metric_value, epoch * len(self.train_dataloader) + i)

                self.logger.info(time_str + " " + loss_str + " " + metric_str)
                self.tb_logger.add_scalar("train/loss_val", losses.val, epoch * len(self.train_dataloader) + i)
                self.tb_logger.add_scalar("train/loss_avg", losses.avg, epoch * len(self.train_dataloader) + i)
                self.tb_logger.add_scalar("train/epoch", epoch, epoch * len(self.train_dataloader) + i)

        return losses.avg, metric_dict

    def _backup_files(self, source_dir, backup_dir, exclude_dirs=['exp', 'temp', 'output']):
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

    def bulid_logger(self, base_dir):
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
        if self.rank == 0:
            tb_logger = SummaryWriter(log_dir=os.path.join(base_dir))
        else:
            tb_logger = None
        return logger, tb_logger

    def count_param(self, _model):
        return sum([x.nelement() for x in _model.parameters()]), sum([x.nelement() for x in _model.parameters() if x.requires_grad is True])
    
    @torch.no_grad()
    def val_epoch(self):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        end = time.time()
        device = next(self.model.parameters()).device
        global_metric_dict = {}
        for i, (images, target) in enumerate(self.val_dataloader):
            # measure data loading time
            data_time.update(time.time() - end)

            # move data to the same device as model
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output
            output = self.model(images)
            loss = self.criterion(output, target)

            batch_size = images.shape[0]
            metric_dict = {}
            for evaluator_name_, evaluator in self.evaluator.items():
                evaluator.update(output, target, batch_size)
                for metric_key, metricAvgMeter in evaluator.metrics.items():
                    metric_dict.update({
                        # metric_key + "_val": metricAvgMeter.val.item(),
                        metric_key + "_avg": metricAvgMeter.avg.item(),
                    })
            losses.update(loss.item(), batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if self.is_distributed:
                # 同步所有 GPU 的损失和指标值
                loss_tensor = torch.tensor([losses.sum, losses.count], dtype=torch.float32, device=device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                losses.avg = loss_tensor[0].item() / loss_tensor[1].item()

                # 同步每个 metric 的值
                for metric_key, metric_value in metric_dict.items():
                    metric_tensor = torch.tensor([metric_value], dtype=torch.float32, device=device)
                    dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
                    global_metric_dict[metric_key] = metric_tensor.item() / dist.get_world_size()
            else:
                global_metric_dict = metric_dict

            if i % self.config.print_freq == 0 and self.rank==0:
                batch_time_val_avg_str = "{:.3f} ({:.3f})".format(batch_time.val, batch_time.avg)
                data_time_val_avg_str = "{:.3f} ({:.3f})".format(data_time.val, data_time.avg)
                loss_val_avg_str = "{:.3f} ({:.3f})".format(losses.val, losses.avg)

                time_str = "Testing: [{} / {}] Batch time: {} Data time: {}".format(i, len(self.val_dataloader), batch_time_val_avg_str, data_time_val_avg_str)

                loss_str = "Loss: {}".format(loss_val_avg_str)
                mertic_str = ""
                for mertic_key, mertic_value in metric_dict.items():
                    mertic_str += "{}: {:.3f}  ".format(mertic_key, mertic_value)

                self.logger.info(time_str + " " + loss_str + " " + mertic_str)

        return {"ce_loss": losses.avg}, global_metric_dict
    
    def log_after_val_epoch(self, loss_dict, metric_dict, epoch=-1):
        if self.rank == 0:
            self.logger.info("******************* Finish validate for epoch {} *******************".format(epoch))
            for loss_key, loss_val in loss_dict.items():
                self.logger.info("The validate loss {} is: {:.4f}".format(loss_key, loss_val))
                self.tb_logger.add_scalar('eval/val_loss', loss_val, epoch)

            for metric_key, metric_val in metric_dict.items():
                self.logger.info("The metric {} is: {:.4f}".format(metric_key, metric_val))
                self.tb_logger.add_scalar('eval/{}'.format(metric_key), metric_val, epoch)
    
    def save_after_val_epoch(self, metric_dict, epoch=-1):
        if self.rank == 0:
            recoder_key = self.recoder_key
            if metric_dict[self.recoder_key] > self.recoder[self.recoder_key]:
                self.recoder[self.recoder_key] = metric_dict[self.recoder_key]
                save_dir = os.path.join(self.output_dir, "ckpt")
                os.makedirs(save_dir, exist_ok=True)
                self.logger.info("************ Get best result {}: {:.4f} in epoch {} ************".format(self.recoder_key, metric_dict[self.recoder_key], epoch))
                self.save_checkpoint({
                    'epoch': epoch + 1,
                    'model': self.config.model,
                    'state_dict': self.model.state_dict(),
                    recoder_key: self.recoder[self.recoder_key],
                    'optimizer' : self.optimizer.state_dict(),
                    'scheduler' : self.scheduler.state_dict()
                }, is_best=True, save_dir=save_dir)
    
    def save_checkpoint(self, state, is_best, save_dir='output', filename='checkpoint.pth.tar'):
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        torch.save(state, save_path)
        if is_best:
            shutil.copyfile(save_path, os.path.join(save_dir, 'model_best.pth.tar'))
    
    def train_step(self, batch):
        pass
    
    def train_log_epoch(self, batch):
        pass

    def train_log_step(self, batch):
        pass

    def val_step(self, batch):
        pass

    def val_log_step(self, batch):
        pass

    def val_log_epoch(self, batch):
        pass