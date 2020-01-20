from archai.nas.arch_trainer import ArchTrainer
from archai.common.common import common_init
from archai.nas import search
from archai.petridish.petridish_micro_builder import PetridishMicroBuilder


if __name__ == '__main__':
    conf = common_init(config_filepath='confs/petridish_cifar.yaml',
                       param_args=['--common.experiment_name', 'petridish_cifar_search'])

    # region config
    conf_search = conf['nas']['search']
    # endregion

    micro_builder = PetridishMicroBuilder()
    trainer_class = ArchTrainer

    search.search_arch(conf_search, micro_builder, trainer_class)

    exit(0)
