import time
import copy
from typing import List, Optional, Tuple
import pathlib
import math
import statistics

from collections import defaultdict
from torch import Tensor

import yaml

from . import utils
from .common import get_logger, get_tb_writer, expdir_abspath

class Metrics:
    """Record top1, top5, loss metrics, track best so far.

    There are 3 levels of metrics:
    1. Run level - these for the one call of 'fit', example, best top1
    2. Epoch level - these are the averages maintained top1, top5, loss
    3. Step level - these are for every step in epoch

    The pre_run must be called before fit call which will reset all metrics. Similarly
    pre_epoch will reset running averages and pre_step will reset step level metrics like average step time.

    The post_step will simply update the running averages while post_epoch updates
    best we have seen for each epoch.
    """

    def __init__(self, title:str, logger_freq:int=10, run_info={}, enable_tb=True) -> None:
        self.logger_freq = logger_freq
        self.title, self.run_info, self.enable_tb = title, run_info, enable_tb
        self._reset_run()

    def _reset_run(self)->None:
        self.run_metrics = RunMetrics()
        self.global_step = -1
        self._tb_path = get_logger().path()

    def pre_run(self, resuming:bool)->None:
        if not resuming:
            self._reset_run()
            self.run_metrics.pre_run()
        else:
            # load_state_dict was called, checkpoint must be done after
            # at least one epoch was completed
            assert self.run_metrics.epoch >= 0

        # logging
        if self.logger_freq > 0:
            logger = get_logger()
            logger.debug({'resuming': resuming, 'run_info':self.run_info})

    def post_run(self)->None:
        self.run_metrics.post_run()

        # logging
        if self.logger_freq > 0:
            logger = get_logger()
            with logger.pushd('timings'):
                logger.info({'epoch':self.run_metrics.epoch_time_avg(),
                            'step': self.run_metrics.step_time_avg(),
                            'run': self.run_metrics.duration()})

            best_train, best_val = self.run_metrics.best_epoch()
            with logger.pushd('best_train'):
                logger.info({'epoch': best_train.index,
                            'top1': best_train.top1.avg})

            if best_val:
                with logger.pushd('best_val'):
                    logger.info({'epoch': best_val.index,
                                'top1': best_val.val_metrics.top1.avg})

    def pre_step(self, x: Tensor, y: Tensor):
        self.run_metrics.cur_epoch().pre_step()
        self.global_step += 1

    def post_step(self, x: Tensor, y: Tensor, logits: Tensor,
                  loss: Tensor, steps: int) -> None:
        # update metrics after optimizer step
        batch_size = x.size(0)
        top1, top5 = utils.accuracy(logits, y, topk=(1, 5))

        epoch = self.run_metrics.cur_epoch()
        epoch.post_step(top1.item(), top5.item(),
                                              loss.item(), batch_size)

        logger = get_logger()
        if self.logger_freq > 0 and \
                (self.global_step+1 % self.logger_freq == 0):
            logger.info({'top1': epoch.top1.avg,
                        'top5': epoch.top5.avg,
                        'loss': epoch.loss.avg})

        if self.enable_tb:
            writer = get_tb_writer()
            writer.add_scalar(f'{self._tb_path}/train_steps/loss',
                                epoch.loss.avg, self.global_step)
            writer.add_scalar(f'{self._tb_path}/train_steps/top1',
                                epoch.top1.avg, self.global_step)
            writer.add_scalar(f'{self._tb_path}/train_steps/top5',
                                epoch.top5.avg, self.global_step)

    def pre_epoch(self, lr:float=math.nan)->None:
        epoch = self.run_metrics.add_epoch()
        epoch.pre_epoch(lr)
        if lr is not None:
            logger, writer = get_logger(), get_tb_writer()
            if self.logger_freq > 0 and not math.isnan(lr):
                logger.debug({'start_lr': lr})
            if self.enable_tb:
                writer.add_scalar(f'{self._tb_path}/train_steps/lr',
                                  lr, self.global_step)

    def post_epoch(self, val_metrics:Optional['Metrics'], lr:float=math.nan):
        epoch = self.run_metrics.cur_epoch()
        epoch.post_epoch(val_metrics, lr)
        test_epoch = None
        if val_metrics:
            test_epoch = val_metrics.run_metrics.epochs_metrics[0]

        if self.logger_freq > 0:
            logger = get_logger()
            with logger.pushd('train'):
                logger.info({'top1': epoch.top1.avg,
                            'top5': epoch.top5.avg,
                            'loss': epoch.loss.avg,
                            'end_lr': lr})
            if test_epoch:
                with logger.pushd('val'):
                    logger.info({'top1': test_epoch.top1.avg,
                                'top5': test_epoch.top5.avg,
                                'loss': test_epoch.loss.avg})

        if self.enable_tb:
            writer = get_tb_writer()
            writer.add_scalar(f'{self._tb_path}/train_epochs/loss',
                                epoch.loss.avg, epoch.index)
            writer.add_scalar(f'{self._tb_path}/train_epochs/top1',
                                epoch.top1.avg, epoch.index)
            writer.add_scalar(f'{self._tb_path}/train_epochs/top5',
                                epoch.top5.avg, epoch.index)
            if test_epoch:
                writer.add_scalar(f'{self._tb_path}/val_epochs/loss',
                                    test_epoch.loss.avg, epoch.index)
                writer.add_scalar(f'{self._tb_path}/val_epochs/top1',
                                    test_epoch.top1.avg, epoch.index)
                writer.add_scalar(f'{self._tb_path}/val_epochs/top5',
                                    test_epoch.top5.avg, epoch.index)

    def state_dict(self)->dict:
        d = utils.state_dict(self)
        assert isinstance(d, dict)
        return d

    def load_state_dict(self, state_dict:dict)->None:
        utils.load_state_dict(self, state_dict)

    def save(self, filename:str)->Optional[str]:
        save_path = expdir_abspath(filename)
        if save_path:
            if not save_path.endswith('.yaml'):
                save_path += '.yaml'
            pathlib.Path(save_path).write_text(yaml.dump(self))
        return save_path

    def epochs(self)->int:
        return len(self.run_metrics.epochs_metrics)

    def cur_epoch(self)->'EpochMetrics':
        return self.run_metrics.cur_epoch()


class Accumulator:
    # TODO: replace this with Metrics class
    def __init__(self):
        self.metrics = defaultdict(lambda: 0.)

    def add(self, key, value):
        self.metrics[key] += value

    def add_dict(self, dict):
        for key, value in dict.items():
            self.add(key, value)

    def __getitem__(self, item):
        return self.metrics[item]

    def __setitem__(self, key, value):
        self.metrics[key] = value

    def get_dict(self):
        return copy.deepcopy(dict(self.metrics))

    def items(self):
        return self.metrics.items()

    def __str__(self):
        return str(dict(self.metrics))

    def __truediv__(self, other):
        newone = Accumulator()
        for key, value in self.items():
            if isinstance(other, str):
                if other != key:
                    newone[key] = value / self[other]
                else:
                    newone[key] = value
            else:
                newone[key] = value / other
        return newone

class EpochMetrics:
    def __init__(self, index:int) -> None:
        self.index = index
        self.top1 = utils.AverageMeter()
        self.top5 = utils.AverageMeter()
        self.loss = utils.AverageMeter()
        self.step_time = utils.AverageMeter()
        self.start_time = math.nan
        self.end_time = math.nan
        self.step = -1
        self.start_lr = math.nan
        self.end_lr = math.nan
        self.val_metrics:Optional[EpochMetrics] = None

    def pre_step(self):
        self._step_start_time = time.time()
        self.step += 1
    def post_step(self, top1:float, top5:float, loss:float, batch:int):
        self.step_time.update(time.time() - self._step_start_time)
        self.top1.update(top1, batch)
        self.top5.update(top5, batch)
        self.loss.update(loss, batch)

    def pre_epoch(self, lr:float):
        self.start_time = time.time()
        self.start_lr = lr
    def post_epoch(self, val_metrics:Optional[Metrics], lr:float):
        self.end_time = time.time()
        self.end_lr = lr
        self.val_metrics = val_metrics.run_metrics.epochs_metrics[-1] \
                                if val_metrics is not None else None
    def duration(self):
        return self.end_time-self.start_time

class RunMetrics:
    def __init__(self) -> None:
        self.epochs_metrics:List[EpochMetrics] = []
        self.start_time = math.nan
        self.end_time = math.nan
        self.epoch = -1

    def pre_run(self):
        self.start_time = time.time()
    def post_run(self):
        self.end_time = time.time()

    def add_epoch(self)->EpochMetrics:
        self.epoch = len(self.epochs_metrics)
        epoch_metrics = EpochMetrics(self.epoch)
        self.epochs_metrics.append(epoch_metrics)
        return epoch_metrics

    def cur_epoch(self)->EpochMetrics:
        return self.epochs_metrics[self.epoch]

    def best_epoch(self)->Tuple[EpochMetrics, Optional[EpochMetrics]]:
        best_train = max(self.epochs_metrics, key=lambda e:e.top1.avg)
        best_val = max(self.epochs_metrics,
            key=lambda e:e.val_metrics.top1.avg if e.val_metrics else -1)
        best_val = best_val if best_val.val_metrics else None
        return best_train, best_val

    def epoch_time_avg(self):
        return statistics.mean((e.duration() for e in self.epochs_metrics))
    def step_time_avg(self):
        return statistics.mean((e.step_time.avg for e in self.epochs_metrics))

    def duration(self):
        return self.end_time-self.start_time