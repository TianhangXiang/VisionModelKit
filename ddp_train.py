import os
import argparse
import torch
import torch.distributed as dist
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

import trainer

def setup(local_rank, local_world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=local_rank, world_size=local_world_size)

def cleanup():
    dist.destroy_process_group()

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
    parser.add_argument('--recoder_key', type=str, default='Acc@1_avg')

    # ddp args
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--local_world_size", type=int, default=1)

    config = parser.parse_args()

    # init process group
    setup(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]))

    # These are the parameters used to initialize the process group
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }
    
    # #TODO: fix logger use
    # logger.info(
    #     "Initializing process group: MASTER_ADDR: {}, MASTER_PORT: {}, RANK: {}, WORLD_SIZE: {}, BACKEND: {}".format(
    #         env_dict["MASTER_ADDR"], env_dict["MASTER_PORT"], config.rank, env_dict["WORLD_SIZE"], dist.get_backend()))
    # logger.debug("Using configuration: \n{}".format(config))
    torch.cuda.set_device(config.local_rank)


    trainer = trainer.BaseTrainer(config=config)
    trainer.train()
    # trainer.val()
    
    


