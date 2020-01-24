from typing import Tuple, Optional

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from overrides import EnforceOverrides

from .metrics import Metrics
from .config import Config
from . import utils
from .common import get_logger

class Tester(EnforceOverrides):
    """Evaluate model on given data
    """

    def __init__(self, conf_eval:Config, model:nn.Module, device,
                 aux_tower:bool)->None:
        self._title = conf_eval['title']
        self._logger_freq = conf_eval['logger_freq']
        conf_lossfn = conf_eval['lossfn']

        self.model = model
        self.device = device
        self._aux_tower = aux_tower
        self._lossfn = utils.get_lossfn(conf_lossfn).to(device)
        self._metrics = None

    def test(self, test_dl: DataLoader)->Metrics:
        logger = get_logger()
        logger.pushd(self._title)

        self._metrics = self._create_metrics()

        # recreate metrics for this run
        self._pre_test(False)
        self._test_epoch(test_dl)
        self._post_test()

        logger.popd()
        return self.get_metrics() # type: ignore

    def _test_epoch(self, test_dl: DataLoader)->None:
        logger = get_logger()

        self._metrics.pre_epoch()
        self.model.eval()
        steps = len(test_dl)

        with torch.no_grad(), logger.pushd('steps'):
            for step, (x, y) in enumerate(test_dl):
                logger.pushd(step)

                assert not self.model.training # derived class might alter the mode

                # enable non-blocking on 2nd part so its ready when we get to it
                x, y = x.to(self.device), y.to(self.device, non_blocking=True)

                self._pre_step(x, y, self._metrics)
                logits = self.model(x)
                if self._aux_tower:
                    logits = logits[0]
                loss = self._lossfn(logits, y)
                self._post_step(x, y, logits, loss, steps, self._metrics)

                logger.popd()
        self._metrics.post_epoch(None)

    def get_metrics(self)->Optional[Metrics]:
        return self._metrics

    def state_dict(self)->dict:
        return {
            'metrics': self._metrics.state_dict()
        }

    def load_state_dict(self, state_dict:dict)->None:
        self._metrics.load_state_dict(state_dict['metrics'])

    def _pre_test(self, resuming:bool)->None:
        self._metrics.pre_run(resuming)

    def _post_test(self)->None:
        self._metrics.post_run()

    def _pre_step(self, x:Tensor, y:Tensor, metrics:Metrics)->None:
        metrics.pre_step(x, y)

    def _post_step(self, x:Tensor, y:Tensor, logits:Tensor, loss:Tensor,
                  steps:int, metrics:Metrics)->None:
        metrics.post_step(x, y, logits, loss, steps)

    def _create_metrics(self)->Metrics:
        return Metrics(self._title, logger_freq=self._logger_freq)

