from collections import OrderedDict
import torch

class AverageMeter(object):
    """From: https://github.com/HobbitLong/CMC/blob/master/util.py"""
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
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

class MultiAverageMeter():
    def __init__(self):
        self.losses = OrderedDict()

    def init_loss(self, name):
        self.losses[name] = AverageMeter()

    def reset(self):
        for k,v in self.losses.items():
            v.reset()

    def update(self, vals):
        '''vals is a dictionary mapping {loss name} -> {val}'''
        for k,v in vals.items():
            if type(v) is torch.Tensor:
                v = v.item()
            self.losses[k].update(v)

    def print(self):
        loss_strings = []
        for k,v in self.losses.items():
            loss_strings.append(f'{k}: {v.avg:0.05f}')

        output = ' \n'.join(loss_strings)
        print('Losses: ')
        print(output)
        print('')
