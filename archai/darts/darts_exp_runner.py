from typing import Type

from ..nas.exp_runner import ExperimentRunner
from .darts_micro_builder import DartsMicroBuilder
from .bilevel_arch_trainer import BilevelArchTrainer

from overrides import overrides

class DartsExperimentRunner(ExperimentRunner):
    @overrides
    def micro_builder(self)->DartsMicroBuilder:
        return DartsMicroBuilder()

    @overrides
    def trainer_class(self)->Type[BilevelArchTrainer]:
        return BilevelArchTrainer

