from typing import Iterable, Tuple, Optional, Any
from collections import OrderedDict
import numpy as np
import yaml

import torch
from torch import nn, Tensor

from overrides import overrides


from .cell import Cell
from .operations import Op, DropPath_
from .model_desc import ModelDesc, AuxTowerDesc, CellDesc
from ..common.common import get_logger, expdir_abspath
from ..common import utils

class Model(nn.Module):
    def __init__(self, model_desc:ModelDesc, droppath:bool, affine:bool):
        super().__init__()

        logger = get_logger()

        self.desc = model_desc
        self._stem0_op = Op.create(model_desc.stem0_op, affine=affine)
        self._stem1_op = Op.create(model_desc.stem1_op, affine=affine)

        self._cells = nn.ModuleList()
        self._aux_towers = nn.ModuleList()

        for i, (cell_desc, aux_tower_desc) in \
                enumerate(zip(model_desc.cell_descs, model_desc.aux_tower_descs)):
            self._build_cell(cell_desc, aux_tower_desc, droppath, affine)

        # adaptive pooling output size to 1x1
        self.final_pooling = Op.create(model_desc.pool_op, affine=affine)
        # since ch_p records last cell's output channels
        # it indicates the input channel number
        self.linear = nn.Linear(model_desc.cell_descs[-1].cell_ch_out,
                                model_desc.n_classes)

        logger.info(f'{yaml.dump(self.summary())}')

    def _build_cell(self, cell_desc:CellDesc,
                    aux_tower_desc:Optional[AuxTowerDesc],
                    droppath:bool, affine:bool)->None:
        assert cell_desc.index == len(self._cells)
        alphas_cell = None if cell_desc.alphas_from==cell_desc.index  \
                            else self._cells[cell_desc.alphas_from]
        cell = Cell(cell_desc,
                    affine=affine, droppath=droppath,
                    alphas_cell=alphas_cell)
        self._cells.append(cell)
        self._aux_towers.append(AuxTower(aux_tower_desc, pool_stride=3) \
                                if aux_tower_desc else None)

    def summary(self)->dict:
        return {
            'cell_count': len(self._cells),
            #'cell_params': [utils.param_size(c) for c in self._cells]
            'params': utils.param_size(self),
            'alphas_p': len(list(a for a in self.alphas())),
            'alphas': np.sum(a.numel() for a in self.alphas()),
            'ops': np.sum(len(n.edges) for c in self.desc.cell_descs for n in c.nodes),
        }

    def alphas(self)->Iterable[nn.Parameter]:
        for cell in self._cells:
            if not cell.shared_alphas:
                for alpha in cell.alphas():
                    yield alpha

    def weights(self)->Iterable[nn.Parameter]:
        for cell in self._cells:
            for w in cell.weights():
                yield w

    @overrides
    def forward(self, x)->Tuple[Tensor, Optional[Tensor]]:
        """
        Runs x through cells with alphas, applies final pooling, send through
            FCs and returns logits.

        in: torch.Size([3, 3, 32, 32])
        stem: torch.Size([3, 48, 32, 32])
        cell: 0 torch.Size([3, 64, 32, 32]) False
        cell: 1 torch.Size([3, 64, 32, 32]) False
        cell: 2 torch.Size([3, 128, 16, 16]) True
        cell: 3 torch.Size([3, 128, 16, 16]) False
        cell: 4 torch.Size([3, 128, 16, 16]) False
        cell: 5 torch.Size([3, 256, 8, 8]) True
        cell: 6 torch.Size([3, 256, 8, 8]) False
        cell: 7 torch.Size([3, 256, 8, 8]) False
        pool:   torch.Size([16, 256, 1, 1])
        linear: [b, 10]
        """

        # TODO: original code has slighly different way of applying stems
        s0 = self._stem0_op(x)
        s1 = self._stem1_op(x)

        logits_aux = None
        for cell, aux_tower in zip(self._cells, self._aux_towers):
            #print(s0.shape, s1.shape, end='')
            s0, s1 = s1, cell.forward(s0, s1)
            #print('\t->\t', s0.shape, s1.shape)

            # TODO: this mimics darts but won't work for multiple aux towers
            if aux_tower is not None and self.training:
                logits_aux = aux_tower(s1)

        # s1 is now the last cell's output
        out = self.final_pooling(s1)
        logits = self.linear(out.view(out.size(0), -1)) # flatten

        return logits, logits_aux

    def device_type(self)->str:
        return next(self.parameters()).device.type

    def finalize(self, to_cpu=True, restore_device=True)->ModelDesc:
        original = self.device_type()
        if to_cpu:
            self.cpu()

        # finalize will create copy of state and this can overflow GPU RAM
        assert self.device_type() == 'cpu'

        cell_descs = [cell.finalize() for cell in self._cells]

        if restore_device:
            self.to(original, non_blocking=True)

        return ModelDesc(stem0_op=self.desc.stem0_op,
                         stem1_op=self.desc.stem1_op,
                         pool_op=self.desc.pool_op,
                         ds_ch=self.desc.ds_ch,
                         n_classes=self.desc.n_classes,
                         cell_descs=cell_descs,
                         aux_tower_descs=self.desc.aux_tower_descs)

    def drop_path_prob(self, p:float):
        """ Set drop path probability
        This will be called externally so any DropPath_ modules get
        new probability. Typically, every epoch we will reduce this probability.
        """
        for module in self.modules():
            if isinstance(module, DropPath_):
                module.p = p

    def save(self, filename:str)->Optional[str]:
        save_path = expdir_abspath(filename)
        if save_path:
            utils.save(self, save_path)
        return save_path


class AuxTower(nn.Module):
    def __init__(self, aux_tower_desc:AuxTowerDesc, pool_stride:int):
        """assuming input size 14x14"""
        # TODO: assert input size
        super().__init__()

        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=pool_stride, padding=0, count_include_pad=False),
            nn.Conv2d(aux_tower_desc.ch_in, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            # TODO: This batchnorm was omitted in orginal implementation due to a typo.
            # nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.linear = nn.Linear(768, aux_tower_desc.n_classes)

    def forward(self, x:torch.Tensor):
        x = self.features(x)
        x = self.linear(x.view(x.size(0), -1))
        return x