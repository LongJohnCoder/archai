import os

from archai.common.common import common_init
from archai.nas.evaluate import eval_arch
from archai.random_arch.random_micro_builder import RandomMicroBuilder

from archai.nas import nas_utils

if __name__ == '__main__':
    conf = common_init(config_filepath='confs/random_cifar.yaml',
                       param_args=['--common.experiment_name', 'random_cifar_eval'])

    # region config
    conf_eval = conf['nas']['eval']
    final_desc_filename = conf_eval['final_desc_filename']
    # endregion

    # evaluate architecture using eval settings
    eval_arch(conf_eval, micro_builder=RandomMicroBuilder())

    exit(0)


# LOG_DIR = os.environ.get('PT_OUTPUT_DIR', '')
# DATA_DIR = os.environ.get('PT_DATA_DIR', '')
