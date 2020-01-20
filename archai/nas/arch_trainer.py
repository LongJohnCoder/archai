from typing import Optional, Callable
import os

import torch
from torch.utils.data import DataLoader
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from overrides import overrides, EnforceOverrides

from ..common.config import Config
from ..common import common
from ..nas.model import Model
from ..nas.model_desc import ModelDesc
from ..common.trainer import Trainer
from ..nas.vis_model_desc import draw_model_desc
from ..common.check_point import CheckPoint

class ArchTrainer(Trainer, EnforceOverrides):
    def __init__(self, conf_train: Config, model: Model, device,
                 check_point:Optional[CheckPoint]) -> None:
        super().__init__(conf_train, model, device, check_point, aux_tower=True)

        self._l1_alphas = conf_train['l1_alphas']
        self._plotsdir = common.expdir_abspath(conf_train['plotsdir'], True)

    @overrides
    def compute_loss(self, lossfn: Callable,
                     x: Tensor, y: Tensor, logits: Tensor,
                     aux_weight: float, aux_logits: Optional[Tensor]) -> Tensor:
        loss = super().compute_loss(lossfn, x, y, logits,
                                    aux_weight, aux_logits)
        if self._l1_alphas > 0.0:
            l_extra = sum(torch.sum(a.abs()) for a in self.model.alphas())
            loss += self._l1_alphas * l_extra
        return loss

    @overrides
    def post_epoch(self, train_dl: DataLoader, val_dl: Optional[DataLoader])->None:
        super().post_epoch(train_dl, val_dl)
        self._draw_model()

    def _draw_model(self) -> None:
        if not self._plotsdir:
            return
        train_metrics, val_metrics = self.get_metrics()
        if (val_metrics and val_metrics.is_best()) or \
                (train_metrics and train_metrics.is_best()):

            # log model_desc as a image
            plot_filepath = os.path.join(
                self._plotsdir, "EP{train_metrics.epoch:03d}")
            draw_model_desc(self.model.finalize(), plot_filepath+"-normal",
                            caption=f"Epoch {train_metrics.epoch}")
