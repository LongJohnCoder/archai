from abc import ABC

from overrides import EnforceOverrides

from .model_desc import ModelDesc

class MicroBuilder(ABC, EnforceOverrides):
    def register_ops(self)->None:
        pass
    def build(self, model_desc:ModelDesc, search_iteration:int)->None:
        pass # prepare model for the given search iteration
    def seed(self, model_desc:ModelDesc)->None:
        pass # prepare model as seed model

