from typing import Mapping, Optional, Union
import copy

import torch
from torch.utils.data import DataLoader
from torch import Tensor, nn, autograd
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from overrides import overrides

from ..common.config import Config
from ..nas.arch_trainer import ArchTrainer
from ..common import utils
from ..nas.model import Model
from ..common.check_point import CheckPoint
from ..common.common import get_logger


class XnasArchTrainer(ArchTrainer):
    def __init__(self, conf_train: Config, model: Model, device,
                 check_point:Optional[CheckPoint]) -> None:
        super().__init__(conf_train, model, device, check_point)

        self._conf_w_optim = conf_train['optimizer']
        self._conf_w_lossfn = conf_train['lossfn']
        self._conf_alpha_optim = conf_train['alpha_optimizer']

    @overrides
    def create_optimizer(self) -> Optimizer:
        # return optim that only operates on w, not alphas
        return utils.create_optimizer(self._conf_w_optim, self.model.weights())

    @overrides
    def pre_fit(self, train_dl: DataLoader, val_dl: Optional[DataLoader],
                resuming:bool) -> None:
        super().pre_fit(train_dl, val_dl, resuming)

        # optimizers, schedulers needs to be recreated for each fit call
        # as they have state
        assert val_dl is not None       
        lossfn = utils.get_lossfn(self._conf_w_lossfn).to(self.device)

        self._xnas_optim = _XnasOptimizer(self._conf_alpha_optim, self.model, lossfn)


    @overrides
    def post_fit(self, train_dl:DataLoader, val_dl:Optional[DataLoader])->None:
        # delete state we created in pre_fit
        del self._xnas_optim
        return super().post_fit(train_dl, val_dl)

    @overrides
    def pre_epoch(self, train_dl: DataLoader, val_dl: Optional[DataLoader])->None:
        super().pre_epoch(train_dl, val_dl)

        # prep val set to train alphas
        self._valid_iter = iter(val_dl)  # type: ignore

    @overrides
    def post_epoch(self, train_dl:DataLoader, val_dl:Optional[DataLoader])->None:
        del self._valid_iter # clean up
        super().post_epoch(train_dl, val_dl)

    @overrides
    def pre_step(self, x: Tensor, y: Tensor) -> None:
        super().pre_step(x, y)

        # reset val loader if we exhausted it
        try:
            x_val, y_val = next(self._valid_iter)
        except StopIteration:
            # reinit iterator
            self._valid_iter = iter(self._val_dl)
            x_val, y_val = next(self._valid_iter)

        x_val, y_val = x_val.to(self.device), y_val.to(
            self.device, non_blocking=True)

        # update alphas
        self._xnas_optim.step(x, y, x_val, y_val)

    @overrides
    def update_checkpoint(self, check_point:CheckPoint)->None:
        super().update_checkpoint(check_point)
       

class _XnasOptimizer:
    def __init__(self, conf_alpha_optim:Config,
                 model: Model, lossfn: _Loss) -> None:
        logger = get_logger()
        self._alpha_lr = conf_alpha_optim['lr']
       
        self._lossfn = lossfn
        self._model = model  # main model with respect to w and alpha

    @staticmethod
    def _get_loss(model, lossfn, x, y):
        logits, *_ = model(x) # might also return aux tower logits
        return lossfn(logits, y)

    def step(self, x_train: Tensor, y_train: Tensor, x_valid: Tensor, y_valid: Tensor) -> None:

        # put model in train mode just to be safe
        self._model.train()

        # put model through val data
        loss = self._get_loss(self._model, self._lossfn, x_valid, y_valid)

        # compute gradients
        loss.backward()

        # for each op in the model update alphas 
        for op in self._model.alphaops():
            op.update_alphas()

            


            
        

        
        

