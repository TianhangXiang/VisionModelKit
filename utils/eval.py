import torch
from enum import Enum
import torch.distributed as dist
import logging

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
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

    # def all_reduce(self):
    #     if torch.cuda.is_available():
    #         device = torch.device("cuda")
    #     else:
    #         device = torch.device("cpu")
    #     total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
    #     dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
    #     self.sum, self.count = total.tolist()
    #     self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    # def summary(self):
    #     fmtstr = ''
    #     if self.summary_type is Summary.NONE:
    #         fmtstr = ''
    #     elif self.summary_type is Summary.AVERAGE:
    #         fmtstr = '{name} {avg:.3f}'
    #     elif self.summary_type is Summary.SUM:
    #         fmtstr = '{name} {sum:.3f}'
    #     elif self.summary_type is Summary.COUNT:
    #         fmtstr = '{name} {count:.3f}'
    #     else:
    #         raise ValueError('invalid summary type %r' % self.summary_type)
        
    #     return fmtstr.format(**self.__dict__)

class Acc():
    def __init__(self) -> None:
        self.meter_top1 = AverageMeter('Acc@1', ':6.2f')
        self.meter_top5 = AverageMeter('Acc@5', ':6.2f')
        self.reset()
        self.metrics = {
            'Acc@1': self.meter_top1,
            'Acc@5': self.meter_top5,
        }

    @torch.no_grad()
    def update(self, pred, gt, batch_size):
        acc_top1, acc_top5 = self.accuracy(pred=pred, target=gt, topk=(1, 5))
        self.meter_top1.update(acc_top1, batch_size)
        self.meter_top5.update(acc_top5, batch_size)

    def reset(self):
        self.meter_top1.reset()
        self.meter_top5.reset()

    def accuracy(self, pred, target, topk=(1, 5)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = pred.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res
    
    def get_values(self):
        pass