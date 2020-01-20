
import  os
from typing import Iterable, Type, MutableMapping, Mapping, Any, Optional, Tuple, List
import  numpy as np
import  shutil
import logging
import csv
from collections import OrderedDict
import sys

import  torch
from torch import nn
from torch.optim import lr_scheduler, SGD, Adam
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.nn.modules.loss import _WeightedLoss, _Loss
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from .config import Config
from .cocob import CocobBackprop

class AverageMeter:

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0.
        self.sum = 0.
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res


class Cutout:
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def param_size(module:nn.Module):
    """count all parameters excluding auxiliary"""
    return np.sum(v.numel() for name, v in module.named_parameters() \
        if "auxiliary" not in name)


def save_checkpoint(model:nn.Module, optim:Optimizer, best_top1:float,
        epoch:int, is_best:bool, ckpt_dir:str)->None:
    state = {
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'best_top1': best_top1,
            'optim': optim.state_dict(),
    }
    filename = os.path.join(ckpt_dir, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(ckpt_dir, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    #logger.info('saved to model: {}'.format(model_path))
    torch.save(model.state_dict(), model_path)

def load(model, model_path):
    #logger.info('load from model: {}'.format(model_path))
    model.load_state_dict(torch.load(model_path))

def drop_path_(x, drop_prob, training):
    if training and drop_prob > 0.:
        keep_prob = 1. - drop_prob
        # Bernoulli returns 1 with pobability p and 0 with 1-p.
        # Below generates tensor of shape (batch,1,1,1) filled with 1s and 0s
        #   as per keep_prob.
        mask = torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob) \
            .to(device=x.device)
        # scale tensor by 1/p as we will be losing other values
        # for each tensor in batch, zero out values with probability p
        x.div_(keep_prob).mul_(mask)
    return x

def first_or_default(it:Iterable, default=None):
    for i in it:
        return i
    return default

def deep_update(d:MutableMapping, u:Mapping, map_type:Type[MutableMapping]=dict)\
        ->MutableMapping:
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = deep_update(d.get(k, map_type()), v)
        else:
            d[k] = v
    return d

def create_optimizer(conf_opt:Config, params)->Optimizer:
    if conf_opt['type'] == 'sgd':
        return SGD(
           params,
            lr=conf_opt['lr'],
            momentum=conf_opt['momentum'],
            weight_decay=conf_opt['decay'],
            nesterov=conf_opt['nesterov']
        )
    elif conf_opt['type'] == 'adam':
         return Adam(params,
            lr=conf_opt['lr'],
            betas=conf_opt['betas'],
            weight_decay=conf_opt['decay'])
    elif conf_opt['type'] == 'cocob':
        return CocobBackprop(params,
            alpha=conf_opt['alpha'])
    else:
        raise ValueError('invalid optimizer type=%s' % conf_opt['type'])

def get_optim_lr(optimizer:Optimizer)->float:
    for param_group in optimizer.param_groups:
        return param_group['lr']
    raise RuntimeError('optimizer did not had any param_group named lr!')

def ensure_pytorch_ver(min_ver:str, error_msg:str)->bool:
    tv = torch.__version__.split('.')
    req = min_ver.split('.')
    for i,j in zip(tv, req):
        i,j = int(i), int(j)
        if i > j:
            return True
        if i < j:
            if error_msg:
                raise RuntimeError(f'Minimum required PyTorch version is {min_ver} but installed version is {torch.__version__}: {error_msg}')
            return False
    return True


def create_lr_scheduler(conf_lrs:Config, epochs:int, optimizer:Optimizer,
        steps_per_epoch:Optional[int])-> Tuple[Optional[_LRScheduler], bool]:

    # epoch_or_step - apply every epoch or every step
    scheduler, epoch_or_step = None, True

    if conf_lrs is not None:
        lr_scheduler_type = conf_lrs['type'] # TODO: default should be none?
        if lr_scheduler_type == 'cosine':
            # adjust max epochs for warmup
            # TODO: shouldn't we be increasing epochs or schedule lr only after warmup?
            if conf_lrs.get('warmup', None):
                epochs -= conf_lrs['warmup']['epochs']
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs,
                eta_min=conf_lrs['min_lr'])
        elif lr_scheduler_type == 'resnet':
            scheduler = _adjust_learning_rate_resnet(optimizer, epochs)
        elif lr_scheduler_type == 'pyramid':
            scheduler = _adjust_learning_rate_pyramid(optimizer, epochs,
                get_optim_lr(optimizer))
        elif lr_scheduler_type == 'step':
            decay_period = conf_lrs['decay_period']
            gamma = conf_lrs['gamma']
            scheduler = lr_scheduler.StepLR(optimizer, decay_period, gamma=gamma)
        elif lr_scheduler_type == 'one_cycle':
            assert steps_per_epoch is not None
            ensure_pytorch_ver('1.3.0', 'LR scheduler OneCycleLR is not available.')
            max_lr = conf_lrs['max_lr']
            epoch_or_step = False
            scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr,
                            epochs=epochs, steps_per_epoch=steps_per_epoch,
                        )  # TODO: other params
        elif not lr_scheduler_type:
                scheduler = None # TODO: check support for this or use StepLR
        else:
            raise ValueError('invalid lr_schduler=%s' % lr_scheduler_type)

        # select warmup for LR schedule
        if conf_lrs.get('warmup', None):
            scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=conf_lrs['warmup']['multiplier'],
                total_epoch=conf_lrs['warmup']['epochs'],
                after_scheduler=scheduler
            )

    return scheduler, epoch_or_step

def _adjust_learning_rate_pyramid(optimizer, max_epoch:int, base_lr:float):
    def _internal_adjust_learning_rate_pyramid(epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = base_lr * (0.1 ** (epoch // (max_epoch * 0.5))) * (0.1 ** (epoch // (max_epoch * 0.75)))
        return lr

    return lr_scheduler.LambdaLR(optimizer, _internal_adjust_learning_rate_pyramid)


def _adjust_learning_rate_resnet(optimizer, epoch):
    """
    Sets the learning rate to the initial LR decayed by 10 every [90, 180, 240] epochs
    Ref: AutoAugment
    """

    if epoch == 90:
        return lr_scheduler.MultiStepLR(optimizer, [30, 60, 80])
    elif epoch == 270:   # autoaugment
        return lr_scheduler.MultiStepLR(optimizer, [90, 180, 240])
    else:
        raise ValueError('invalid epoch=%d for resnet scheduler' % epoch)

class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets:torch.Tensor, n_classes:int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                    device=targets.device) \
                .fill_(smoothing /(n_classes-1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1.-smoothing)
        return targets

    def forward(self, inputs, targets):
        targets = SmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
            self.smoothing)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss

def get_lossfn(conf_lossfn:Config)->_Loss:
    type = conf_lossfn['type']
    if type == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss()
    elif type == 'CrossEntropyLabelSmooth':
        return SmoothCrossEntropyLoss(smoothing=conf_lossfn['smoothing'])
    else:
        raise ValueError('criterian type "{}" is not supported'.format(type))

def state_dict(val):
    d = getattr(val, '__dict__', None)
    if d is None:
        return val # not an object

    res = {}
    for k,v in val.__dict__.items():
        res[k] = state_dict(v)
    return res

def load_state_dict(val, d:dict)->bool:
    dd = getattr(val, '__dict__', None)
    if dd is None:
        return False

    for k,v in d.items():
        if isinstance(v, dict):
            if load_state_dict(getattr(val, k), v):
               continue
        setattr(val, k, v)
    return True

def deep_comp(o1:Any, o2:Any)->bool:
    # NOTE: dict don't have __dict__
    o1d = getattr(o1, '__dict__', None)
    o2d = getattr(o2, '__dict__', None)

    # if both are objects
    if o1d is not None and o2d is not None:
        # we will compare their dictionaries
        o1, o2 = o1.__dict__, o2.__dict__

    if o1 is not None and o2 is not None:
        # if both are dictionaries, we will compare each key
        if isinstance(o1, dict) and isinstance(o2, dict):
            for k in set().union(o1.keys() ,o2.keys()):
                if k in o1 and k in o2:
                    if not deep_comp(o1[k], o2[k]):
                        return False
                else:
                    return False # some key missing
            return True
    # mismatched object types or both are scalers, or one or both None
    return o1 == o2


# TODO: replace this with SmoothCrossEntropyLoss class
# def cross_entropy_smooth(input: torch.Tensor, target, size_average=True, label_smoothing=0.1):
#     y = torch.eye(10).to(input.device)
#     lb_oh = y[target]

#     target = lb_oh * (1 - label_smoothing) + 0.5 * label_smoothing

#     logsoftmax = nn.LogSoftmax()
#     if size_average:
#         return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
#     else:
#         return torch.sum(torch.sum(-target * logsoftmax(input), dim=1))


def is_debugging()->bool:
    return 'pydevd' in sys.modules # works for vscode

def full_path(path:str)->str:
    path = os.path.expandvars(path)
    path = os.path.expanduser(path)
    return os.path.abspath(path)

def setup_logging(filepath:Optional[str]=None,
                  name:Optional[str]=None, level=logging.INFO)->logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.propagate = False # otherwise root logger prints things again

    if filepath:
        fh = logging.FileHandler(filename=full_path(filepath))
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

def setup_cuda(seed):
    # setup cuda
    cudnn.enabled = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True
    #cudnn.deterministic = False
    # torch.cuda.empty_cache()
    # torch.cuda.synchronize()

def cuda_device_names()->str:
    return ', '.join([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

def append_csv_file(filepath:str, new_row:List[Tuple[str, Any]], delimiter='\t'):
    fieldnames, rows = [], []
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            dr = csv.DictReader(f, delimiter=delimiter)
            fieldnames = dr.fieldnames
            rows = [row for row in dr.reader]
    if fieldnames is None:
        fieldnames = []

    new_fieldnames = OrderedDict([(fn, None) for fn, v in new_row])
    for fn in fieldnames:
        new_fieldnames[fn]=None

    with open(filepath, 'w') as f:
        dr = csv.DictWriter(f, fieldnames=new_fieldnames.keys(), delimiter=delimiter)
        dr.writeheader()
        for row in rows:
            dr.writerow(dict((k,v) for k,v in zip(fieldnames, row)))
        dr.writerow(OrderedDict(new_row))
