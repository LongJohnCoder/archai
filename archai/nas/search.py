
from typing import Type

import torch

from ..common.common import get_logger
from ..common.config import Config
from .micro_builder import MicroBuilder
from .arch_trainer import ArchTrainer
from . import nas_utils
from .model_desc import ModelDesc
from ..common.trainer import Trainer
from ..common import data

def search_arch(conf_search:Config, micro_builder:MicroBuilder,
                trainer_class:Type[ArchTrainer])->None:
    logger = get_logger()

    # region config vars
    conf_model_desc = conf_search['model_desc']
    conf_loader = conf_search['loader']
    conf_train = conf_search['trainer']
    conf_checkpoint = conf_search['checkpoint']
    resume = conf_search['resume']
    final_desc_filename = conf_search['final_desc_filename']
    full_desc_filename = conf_search['full_desc_filename']
    iterations = conf_search['iterations']
    conf_pretrain = conf_search['pretrain']
    # endregion

    device = torch.device(conf_search['device'])

    # if checkpoint is available, load it
    # TODO: search checkpoint is disabled until we figure out
    #       checkpointing for hierarchical iterations
    checkpoint = None # nas_utils.create_checkpoint(conf_checkpoint, resume)
    start_iteration = 0
    if checkpoint is not None and 'search' in checkpoint:
        start_iteration = checkpoint['search']['iteration']
        search_model_desc = ModelDesc.load(full_desc_filename)
    else:
        # create model desc for search using model config
        # we will build model without call to micro_builder for pre-training
        search_model_desc = nas_utils.create_macro_desc(conf_model_desc,
                                    aux_tower=False, template_model_desc=None)

    found_model_desc = None
    assert iterations
    for search_iteration in range(start_iteration, iterations):
        search_model_desc = _pretrained_model_desc(conf_pretrain, device,
                                                   search_model_desc)

        nas_utils.build_micro(search_model_desc, micro_builder, search_iteration)

        # save model desc to enable checkpointing
        search_model_desc.save(full_desc_filename)

        model = nas_utils.model_from_desc(search_model_desc, device,
                                         droppath=False, affine=False)

        # get data
        train_dl, val_dl, _ = data.get_data(conf_loader)
        assert train_dl is not None

        # search arch
        arch_trainer = trainer_class(conf_train, model, device, checkpoint)
        arch_trainer.fit(train_dl, val_dl)

        train_metrics, val_metrics = arch_trainer.get_metrics()
        train_metrics.save('search_train_metrics')
        if val_metrics:
            val_metrics.save('search_val_metrics')

        # save found model
        search_model_desc = model.finalize(restore_device=False)

    found_model_desc = search_model_desc
    save_path = found_model_desc.save(final_desc_filename)
    if save_path:
        logger.info(f"Best architecture saved in {save_path}")
    else:
        logger.info("Best architecture is not saved because file path config not set")

def _pretrained_model_desc(conf_pretrain:Config, device,
                           search_model_desc:ModelDesc)->ModelDesc:
    # region conf vars
    conf_trainer = conf_pretrain['trainer']
    conf_loader = conf_pretrain['loader']
    epochs = conf_trainer['epochs']
    # endregion

    if epochs == 0 or search_model_desc.all_empty():
        # nothing to pretrain
        return search_model_desc
    elif search_model_desc.all_full():
        model = nas_utils.model_from_desc(search_model_desc, device,
                                        droppath=False, affine=True)

        # get data
        train_dl, _, test_dl = data.get_data(conf_loader)
        assert train_dl is not None and test_dl is not None

        trainer = Trainer(conf_trainer, model, device,
            check_point=None, aux_tower=True)
        trainer.fit(train_dl, test_dl)

        # save metrics
        train_metrics, test_metrics = trainer.get_metrics()
        train_metrics.save('search_pretrain_metrics')
        if test_metrics:
            test_metrics.save('search_pretrain_metrics')

        return model.finalize(restore_device=False)
    else:
        raise RuntimeError('Model desc must be full or empty for pretraining')
