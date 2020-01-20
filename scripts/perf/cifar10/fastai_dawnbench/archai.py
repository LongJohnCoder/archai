from typing import Optional

import torch

from archai.common.trainer import Trainer
from archai.common.check_point import CheckPoint
from archai.common.config import Config
from archai.common.common import get_logger, common_init
from archai.common import data
from dawn_net import DawnNet
from basic_net import BasicNet

Net = DawnNet

def train_test(conf_eval:Config):
    logger = get_logger()

    # region conf vars
    conf_loader       = conf_eval['loader']
    save_filename    = conf_eval['save_filename']
    conf_checkpoint = conf_eval['checkpoint']
    resume = conf_eval['resume']
    conf_train = conf_eval['trainer']
    # endregion

    device = torch.device(conf_eval['device'])
    checkpoint = CheckPoint(conf_checkpoint, resume) if conf_checkpoint is not None else None
    model = Net().to(device)

    # get data
    train_dl, _, test_dl = data.get_data(conf_loader)
    assert train_dl is not None and test_dl is not None


    trainer = Trainer(conf_train, model, device, checkpoint, False)
    trainer.fit(train_dl, test_dl)


if __name__ == '__main__':
    conf = common_init(config_filepath='confs/dawnbench.yaml',
                       param_args=['--common.experiment_name', 'dawn_net'])

    conf_eval = conf['nas']['eval']

    # evaluate architecture using eval settings
    train_test(conf_eval)

    exit(0)

