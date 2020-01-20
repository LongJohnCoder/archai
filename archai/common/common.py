import logging
import numpy as np
import os
from typing import List, Iterable, Union, Optional

import yaml

import torch
from torch.utils.tensorboard import SummaryWriter

from .config import Config
from .stopwatch import StopWatch
from . import utils

class SummaryWriterDummy:
    def __init__(self, log_dir):
        pass

    def add_scalar(self, *args, **kwargs):
        pass

SummaryWriterAny = Union[SummaryWriterDummy, SummaryWriter]
_logger: Optional[logging.Logger] = None
_tb_writer: SummaryWriterAny = None


def get_conf()->Config:
    return Config.get()

def get_conf_common()->Config:
    return get_conf()['common']

def get_conf_dataset()->Config:
    return get_conf()['dataset']

def get_experiment_name()->str:
    return get_conf_common()['experiment_name']

def get_expdir()->Optional[str]:
    return get_conf_common()['expdir']

def get_logger() -> logging.Logger:
    global _logger
    if _logger is None:
        raise RuntimeError('get_logger call made before logger was setup!')
    return _logger



def get_tb_writer() -> SummaryWriterAny:
    global _tb_writer
    return _tb_writer

# initializes random number gen, debugging etc
def common_init(config_filepath: Optional[str]=None,
                param_args: list = [],
                log_level=logging.DEBUG, is_master=True, use_args=True) \
        -> Config:

    # TODO: figure out the best way to do this
    datadir = os.environ.get('PT_DATA_DIR', '')
    logdir = os.environ.get('PT_OUTPUT_DIR', '')
    if datadir and logdir:
        # prepend so if supplied from outside it takes back seat
        param_args = ['--nas.eval.loader.dataset.dataroot', datadir,
                      '--nas.search.loader.dataset.dataroot', datadir,
                      '--common.logdir', logdir] + param_args
        print(f'Obtained PT_DATA_DIR {datadir}')
        print(f'Obtained PT_OUTPUT_DIR {logdir}')


    conf = Config(config_filepath=config_filepath,
                  param_args=param_args,
                  use_args=use_args)
    Config.set(conf)

    sw = StopWatch()
    StopWatch.set(sw)

    expdir = _setup_dirs()
    _setup_logger()

    logger = get_logger()
    logger.info(f'expdir: {expdir}')

    _setup_gpus()

    if expdir:
        # copy net config to experiment folder for reference
        with open(os.path.join(expdir, 'full_config.yaml'), 'w') as f:
            yaml.dump(conf, f, default_flow_style=False)

    global _tb_writer
    _tb_writer = _create_tb_writer(is_master)

    return conf

def expdir_abspath(subpath:Optional[str], ensure_exists=False)->Optional[str]:
    """Returns full path for given relative path within experiment directory.
       If experiment directory is not setup then None is returned.
    """

    expdir = get_expdir()
    if not expdir or not subpath:
        return None
    if subpath:
        expdir = os.path.join(expdir, subpath)
        if ensure_exists:
            os.makedirs(expdir, exist_ok=True)

    return expdir

def _create_tb_writer(is_master=True)-> SummaryWriterAny:
    tbdir = expdir_abspath('tb')
    conf_common = get_conf_common()

    WriterClass = SummaryWriterDummy if not conf_common['enable_tb'] or \
                                        not is_master or \
                                        not tbdir \
        else SummaryWriter

    return WriterClass(log_dir=tbdir)

def _setup_dirs()->Optional[str]:
    conf_common = get_conf_common()
    conf_data = get_conf_dataset()
    experiment_name = get_experiment_name()

    # make sure dataroot exists
    dataroot = utils.full_path(conf_data['dataroot'])
    os.makedirs(dataroot, exist_ok=True)

    # make sure logdir and expdir exists
    logdir = conf_common['logdir']
    if logdir:
        logdir = utils.full_path(os.path.expandvars(logdir))
        expdir = os.path.join(logdir, experiment_name)
        os.makedirs(expdir, exist_ok=True)
    else:
        expdir = ''

    # update conf so everyone gets expanded full paths from here on
    conf_common['logdir'], conf_data['dataroot'], conf_common['expdir'] = \
        logdir, dataroot, expdir

    return expdir

def _setup_logger():
    experiment_name = get_experiment_name()

    # file where logger would log messages
    log_filepath = expdir_abspath('logs.log')
    global _logger
    _logger = utils.setup_logging(filepath=log_filepath, name=experiment_name)
    if not log_filepath:
        _logger.warn(
            'logdir not specified, no logs will be created or any models saved')

def _setup_gpus():
    conf_common = get_conf_common()
    logger = get_logger()

    if conf_common['gpus'] is not None:
        csv = str(conf_common['gpus'])
        #os.environ['CUDA_VISIBLE_DEVICES'] = str(conf_common['gpus'])
        torch.cuda.set_device(int(csv.split(',')[0]))
        logger.info('Only these GPUs will be used: {}'.format(
            conf_common['gpus']))
        # alternative: torch.cuda.set_device(config.gpus[0])

    seed = conf_common['seed']
    utils.setup_cuda(seed)

    if conf_common['detect_anomaly']:
        logger.warn(
            'PyTorch code will be 6X slower because detect_anomaly=True.')
        torch.autograd.set_detect_anomaly(True)

    logger.info('Machine has {} gpu(s): {}'.format(torch.cuda.device_count(),
        utils.cuda_device_names()))
    logger.info('Original CUDA_VISIBLE_DEVICES: {}'.format(
        os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ else 'NotSet'))

    # gpu_usage = os.popen(
    #     'nvidia-smi --query-gpu=memory.total,memory.used --format=csv,nounits,noheader'
    # ).read().split('\n')
    # for i, line in enumerate(gpu_usage):
    #     vals = line.split(',')
    #     if len(vals) == 2:
    #         logger.info('GPU {} mem: {}, used: {}'.format(i, vals[0], vals[1]))

