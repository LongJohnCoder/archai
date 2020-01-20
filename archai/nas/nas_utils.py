from typing import Tuple, Optional

from torch.utils.data.dataloader import DataLoader

from .model_desc import ModelDesc
from .macro_builder import MacroBuilder
from .micro_builder import MicroBuilder
from ..common.config import Config
from .model import Model
from ..common.common import get_logger
from ..common.check_point import CheckPoint

def build_micro(model_desc, micro_builder: MicroBuilder, search_iteration:int)->None:
    if micro_builder:
        micro_builder.register_ops()
        micro_builder.build(model_desc, search_iteration)

def create_macro_desc(conf_model_desc: Config, aux_tower:bool,
                      template_model_desc:Optional[ModelDesc])->ModelDesc:
    builder = MacroBuilder(conf_model_desc,
                            aux_tower=aux_tower,
                            template=template_model_desc)
    model_desc = builder.model_desc()
    return model_desc


def create_checkpoint(conf_checkpoint:Config, resume:bool)->Optional[CheckPoint]:
    logger = get_logger()
    checkpoint = CheckPoint(conf_checkpoint, resume) \
                 if conf_checkpoint is not None else None
    if checkpoint is None or checkpoint.is_empty():
        logger.info('Checkpoint not found or resume=False, starting from scratch')
    return checkpoint

def model_and_checkpoint(conf_checkpoint:Config, resume:bool,
        full_desc_filename:str, conf_model_desc: Config,
        device, aux_tower:bool, affine:bool, droppath:bool,
        template_model_desc:ModelDesc) \
                     ->Tuple[Model, Optional[CheckPoint]]:
    logger = get_logger()
    checkpoint = create_checkpoint(conf_checkpoint, resume)
    if checkpoint is None or checkpoint.is_empty():
        # create model
        model_desc = create_macro_desc(conf_model_desc, aux_tower,
                                       template_model_desc)
        model_desc.save(full_desc_filename) # save copy of full model desc
    else:
        logger.info('Checkpoint found, loading last model')
        model_desc = ModelDesc.load(full_desc_filename)

    model = model_from_desc(model_desc, device,
                            droppath=droppath, affine=affine)

    return model, checkpoint

def model_from_desc(model_desc, device, droppath:bool, affine:bool)->Model:
    model = Model(model_desc, droppath=droppath, affine=affine)
    # TODO: enable DataParallel
    # if data_parallel:
    #     model = nn.DataParallel(model).to(device)
    # else:
    model = model.to(device)
    return model

