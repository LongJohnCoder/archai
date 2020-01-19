from FastAutoAugment.nas.model_desc import ModelDesc
from FastAutoAugment.common.common import common_init
from FastAutoAugment.nas.model import Model
from FastAutoAugment.petridish.petridish_micro_builder import PetridishMicroBuilder
from FastAutoAugment.nas.nas_utils import create_macro_desc

from FastAutoAugment.common.model_summary import summary

conf = common_init(config_filepath='confs/petridish_cifar.yaml',
                    param_args=['--common.experiment_name', 'petridish_run2_seed42_eval'])

conf_eval = conf['nas']['eval']
conf_model_desc   = conf_eval['model_desc']

conf_model_desc['n_cells'] = 14
template_model_desc = ModelDesc.load('final_model_desc.yaml')
model_desc = create_macro_desc(conf_model_desc, True, template_model_desc)

mb = PetridishMicroBuilder()
mb.register_ops()
model = Model(model_desc, droppath=False, affine=False)
#model.cuda()
summary(model, [64, 3, 32, 32])


exit(0)