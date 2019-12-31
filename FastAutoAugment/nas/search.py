import os
from typing import Optional, Tuple, TypeVar, Type

import torch

from ..common.common import get_logger, logdir_abspath
from ..common.config import Config
from .model_desc import ModelDesc, RunMode
from .dag_mutator import DagMutator
from .arch_trainer import ArchTrainer
from . import nas_utils

def search_arch(conf_search:Config, dag_mutator:DagMutator,
                trainer_class:Type[ArchTrainer])->None:
    logger = get_logger()

    conf_model_desc = conf_search['model_desc']
    model_desc_filename = conf_search['model_desc_file']
    conf_loader = conf_search['loader']
    conf_train = conf_search['trainer']

    device = torch.device(conf_search['device'])

    # create model
    model = nas_utils.create_model(conf_model_desc, device,
                                   run_mode=RunMode.Search,
                                   dag_mutator=dag_mutator)

    # get data
    train_dl, val_dl = nas_utils.get_train_test_data(conf_loader)

    # search arch
    arch_trainer = trainer_class(conf_train, model, device)
    arch_trainer.fit(train_dl, val_dl)
    found_model_desc = arch_trainer.get_model_desc()

    # save found model
    save_path = nas_utils.save_model_desc(model_desc_filename, found_model_desc)
    if save_path:
        logger.info(f"Best architecture saved in {save_path}")
    else:
        logger.info("Best architecture is not saved because file path config not set")

