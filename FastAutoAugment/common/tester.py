from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from overrides import EnforceOverrides

from .metrics import Metrics
from .config import Config
from . import utils

class Tester(EnforceOverrides):
    """Evaluate model on given data"""

    def __init__(self, conf_eval:Config, model:nn.Module, device)->None:
        self._title = conf_eval['title']
        self._logger_freq = conf_eval['logger_freq']
        conf_lossfn = conf_eval['lossfn']

        self.model = model
        self.device = device
        self._lossfn = utils.get_lossfn(conf_lossfn).to(device)
        self._metrics = self._create_metrics(epochs=1)

    def test(self, test_dl: DataLoader)->None:
        # recreate metrics for this run
        steps = len(test_dl)
        self.pre_test(test_dl, steps, self._metrics)
        self.model.eval()
        with torch.no_grad():
            for x, y in test_dl:
                assert not self.model.training # derived class might alter the mode

                # enable non-blocking on 2nd part so its ready when we get to it
                x, y = x.to(self.device), y.to(self.device, non_blocking=True)

                self.pre_step(x, y, self._metrics)
                logits, *_ = self.model(x) # ignore aux logits in test mode
                loss = self._lossfn(logits, y)
                self.post_step(x, y, logits, loss, steps, self._metrics)
        self.post_test(test_dl, steps, self._metrics)

    def get_metrics(self)->Metrics:
        return self._metrics

    def pre_test(self, test_dl:DataLoader, epoch_steps:int, metrics:Metrics)->None:
        metrics.pre_run(False)
        metrics.pre_epoch()
    def post_test(self, test_dl:DataLoader, epoch_steps:int, metrics:Metrics)->None:
        metrics.post_epoch()
        metrics.post_run()
    def pre_step(self, x:Tensor, y:Tensor, metrics:Metrics)->None:
        metrics.pre_step(x, y)
    def post_step(self, x:Tensor, y:Tensor, logits:Tensor, loss:Tensor,
                  steps:int, metrics:Metrics)->None:
        metrics.post_step(x, y, logits, loss, steps)

    def _create_metrics(self, epochs:int):
        return Metrics(self._title, epochs, logger_freq=self._logger_freq)

