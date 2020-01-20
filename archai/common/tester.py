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

    def __init__(self, conf_eval:Config, model:nn.Module, device,
                 aux_tower:bool, epochs:int=1)->None:
        self._title = conf_eval['title']
        self._logger_freq = conf_eval['logger_freq']
        conf_lossfn = conf_eval['lossfn']

        self.model = model
        self.device = device
        self._aux_tower = aux_tower
        self._lossfn = utils.get_lossfn(conf_lossfn).to(device)
        self._metrics = self._create_metrics(epochs)

    def test(self, test_dl: DataLoader)->None:
        # recreate metrics for this run
        self.pre_test(False)
        self.test_epoch(test_dl)
        self.post_test()

    def test_epoch(self, test_dl: DataLoader)->None:
        self._metrics.pre_epoch()
        self.model.eval()
        steps = len(test_dl)
        with torch.no_grad():
            for x, y in test_dl:
                assert not self.model.training # derived class might alter the mode

                # enable non-blocking on 2nd part so its ready when we get to it
                x, y = x.to(self.device), y.to(self.device, non_blocking=True)

                self.pre_step(x, y, self._metrics)
                logits = self.model(x)
                if self._aux_tower:
                    logits = logits[0]
                loss = self._lossfn(logits, y)
                self.post_step(x, y, logits, loss, steps, self._metrics)
        self._metrics.post_epoch()

    def get_metrics(self)->Metrics:
        return self._metrics

    def state_dict(self)->dict:
        return {
            'metrics': self._metrics.state_dict()
        }

    def load_state_dict(self, state_dict:dict)->None:
        self._metrics.load_state_dict(state_dict['metrics'])

    def increment_epoch(self):
        self._metrics.increment_epoch()

    def pre_test(self, resuming:bool)->None:
        self._metrics.pre_run(resuming)
    def post_test(self)->None:
        self._metrics.post_run()
    def pre_step(self, x:Tensor, y:Tensor, metrics:Metrics)->None:
        metrics.pre_step(x, y)
    def post_step(self, x:Tensor, y:Tensor, logits:Tensor, loss:Tensor,
                  steps:int, metrics:Metrics)->None:
        metrics.post_step(x, y, logits, loss, steps)

    def _create_metrics(self, epochs:int):
        return Metrics(self._title, epochs, logger_freq=self._logger_freq)

