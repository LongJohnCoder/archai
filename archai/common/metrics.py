import time
import copy
from typing import Optional
import pathlib

from collections import defaultdict
from torch import Tensor

import yaml

from . import utils
from .common import get_logger, get_tb_writer, expdir_abspath

class Metrics:
    """Record top1, top5, loss metrics, track best so far"""

    def __init__(self, title:str, epochs: int, logger_freq:int=10) -> None:
        self.logger_freq = logger_freq
        self.title, self.epochs = title, epochs

        self.top1 = utils.AverageMeter()
        self.top5 = utils.AverageMeter()
        self.loss = utils.AverageMeter()
        self.step_time = utils.AverageMeter()
        self.epoch_time = utils.AverageMeter()
        self.run_time = utils.AverageMeter()
        self.custom = {} # user maintained metrics that would be saved/restored

        self.reset()

    def reset(self, epochs:Optional[int]=None)->None:
        self.best_top1, self.best_epoch = 0.0, 0
        self.epoch = 0
        self.global_step = 0
        if epochs is not None:
            self.epochs = epochs

        self.step_time.reset()
        self.epoch_time.reset()
        self._reset_epoch()

    def _reset_epoch(self)->None:
        self.top1.reset()
        self.top5.reset()
        self.loss.reset()
        self.step = 0

    def pre_run(self, resuming:bool)->None:
        if not resuming:
            self.reset()
            self._start_time = time.time()
        assert hasattr(self, '_start_time')
    def post_run(self)->None:
        self._end_time = time.time()
        self.run_time.update(self._end_time-self._start_time)

        if self.logger_freq > 0:
            self.report_times()
            self.report_best()
    def pre_step(self, x: Tensor, y: Tensor):
        self._step_start_time = time.time()

    def post_step(self, x: Tensor, y: Tensor, logits: Tensor,
                  loss: Tensor, steps: int) -> None:
        self._step_end_time = time.time()
        self.step_time.update(self._step_end_time-self._step_start_time)

        # update metrics after optimizer step
        batch_size = x.size(0)
        prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
        self.loss.update(loss.item(), batch_size)
        self.top1.update(prec1.item(), batch_size)
        self.top5.update(prec5.item(), batch_size)
        self.step += 1
        self.global_step += 1

        self.report_cur(steps)

    def report_times(self)->None:
        logger = get_logger()
        logger.info(f'[{self.title}]  Times: {self.epoch_time.avg:.1f} s/epoch, '
                    f'{self.step_time.avg:.1f} s/step '
                    f'{self.run_time.avg:.1f} s/run')

    def increment_epoch(self):
        self.epoch += 1

    def report_cur(self, steps: int):
        if self.logger_freq > 0 and \
                (self.step % self.logger_freq == 0 or \
                 self.step == steps):
            logger = get_logger()
            logger.info(
                f"[{self.title}] "
                f"Epoch: [{self.epoch+1:3d}/{self.epochs}] "
                f"Step {self.step:03d}/{steps:03d} "
                f"Loss {self.loss.avg:.3f} "
                f"Prec@(1,5) ({self.top1.avg:.1%},"
                f" {self.top5.avg:.1%})")

        writer = get_tb_writer()
        writer.add_scalar(f'{self.title}/loss',
                            self.loss.avg, self.global_step)
        writer.add_scalar(f'{self.title}/top1',
                            self.top1.avg, self.global_step)
        writer.add_scalar(f'{self.title}/top5',
                            self.top5.avg, self.global_step)

    def report_best(self):
        if self.logger_freq > 0:
            logger = get_logger()
            logger.info(f"[{self.title}] "
                        f"Final best top1={self.best_top1}, "
                        f"epoch={self.best_epoch}")

    def pre_epoch(self, lr:Optional[float]=None)->None:
        self._reset_epoch()
        self._epoch_start_time = time.time()
        if lr is not None:
            logger, writer = get_logger(), get_tb_writer()
            if self.logger_freq > 0:
                logger.info(f"[{self.title}] Epoch: {self.epoch+1} LR {lr}")
            writer.add_scalar(f'{self.title}/lr', lr, self.global_step)

    def post_epoch(self):
        self._epoch_end_time = time.time()
        self.epoch_time.update(self._epoch_end_time-self._epoch_start_time)
        self.increment_epoch()

        if self.best_top1 < self.top1.avg:
            self.best_epoch = self.epoch
            self.best_top1 = self.top1.avg

    def is_best(self) -> bool:
        return self.epoch == self.best_epoch

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
