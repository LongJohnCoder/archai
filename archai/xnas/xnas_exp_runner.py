from typing import Type

from ..nas.exp_runner import ExperimentRunner
from .xnas_micro_builder import XnasMicroBuilder
from .xnas_arch_trainer import XnasArchTrainer

from overrides import overrides

class XnasExperimentRunner(ExperimentRunner):
    @overrides
    def micro_builder(self)->XnasMicroBuilder:
        return XnasMicroBuilder()

    @overrides
    def trainer_class(self)->Type[XnasArchTrainer]:
        return XnasArchTrainer

