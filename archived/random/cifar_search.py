from archai.common.common import common_init
from archai.random_arch.random_micro_builder import RandomMicroBuilder
from archai.nas import nas_utils

if __name__ == '__main__':
    conf = common_init(config_filepath='confs/random_cifar.yaml',
                       param_args=['--common.experiment_name', 'random_cifar_search'])

    # region config
    conf_search = conf['nas']['search']
    conf_model_desc = conf_search['model_desc']
    final_desc_filename = conf_search['final_desc_filename']
    # endregion

    # create model and save it to yaml
    # NOTE: there is no search here as the models are just randomly sampled
    model_desc = nas_utils.create_macro_desc(conf_model_desc,
                                             aux_tower=False,
                                             template_model_desc=None)
    macro_builder = RandomMicroBuilder()
    macro_builder.build(model_desc, 0)

    # save model to location specified by eval config
    model_desc.save(final_desc_filename)

    exit(0)