from typing import Type

from ..nas.exp_runner import ExperimentRunner
from .petridish_micro_builder import PetridishMicroBuilder
from ..nas.arch_trainer import ArchTrainer

from overrides import overrides

class PetridishExperimentRunner(ExperimentRunner):
    @overrides
    def micro_builder(self)->PetridishMicroBuilder:
        return PetridishMicroBuilder()

    @overrides
    def trainer_class(self)->Type[ArchTrainer]:
        return ArchTrainer

