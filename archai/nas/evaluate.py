from typing import Optional

import torch

from ..common.trainer import Trainer
from ..common.config import Config
from ..common.common import get_logger
from ..common import data
from .model_desc import ModelDesc
from .micro_builder import MicroBuilder
from . import nas_utils

def eval_arch(conf_eval:Config, micro_builder:Optional[MicroBuilder]):
    logger = get_logger()

    # region conf vars
    conf_loader       = conf_eval['loader']
    save_filename    = conf_eval['save_filename']
    conf_model_desc   = conf_eval['model_desc']
    conf_checkpoint = conf_eval['checkpoint']
    resume = conf_eval['resume']
    conf_train = conf_eval['trainer']
    final_desc_filename = conf_eval['final_desc_filename']
    full_desc_filename = conf_eval['full_desc_filename']
    # endregion

    # load model desc file to get template model
    template_model_desc = ModelDesc.load(final_desc_filename)

    device = torch.device(conf_eval['device'])

    if micro_builder:
        micro_builder.register_ops()
    model, checkpoint = nas_utils.model_and_checkpoint(
                                conf_checkpoint, resume, full_desc_filename,
                                conf_model_desc, device,
                                aux_tower=True,
                                affine=True, droppath=True,
                                template_model_desc=template_model_desc)

    # get data
    train_dl, _, test_dl = data.get_data(conf_loader)
    assert train_dl is not None and test_dl is not None


    trainer = Trainer(conf_train, model, device, checkpoint, aux_tower=True)
    trainer.fit(train_dl, test_dl)

    # save metrics
    train_metrics, test_metrics = trainer.get_metrics()
    train_metrics.save('eval_train_metrics')
    if test_metrics:
        test_metrics.save('eval_test_metrics')

    # save model
    save_path = model.save(save_filename)
    if save_path:
        logger.info(f"Model saved in {save_path}")
    else:
        logger.info("Model is not saved because file path config not set")







