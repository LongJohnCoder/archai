import torch
from archai.nas.model import Model
from archai.nas.macro_builder import MacroBuilder
from archai.common.common import common_init

def test_zero_model():
    conf = common_init(config_filepath='confs/darts_cifar.yaml')
    conf_search = conf['nas']['search']
    model_desc = conf_search['model_desc']

    macro_builder = MacroBuilder(model_desc, aux_tower=False)
    model_desc = macro_builder.model_desc()
    m = Model(model_desc, False, True)
    y, aux = m(torch.rand((1, 3, 32, 32)))
    assert isinstance(y, torch.Tensor) and y.shape==(1,10) and aux is None