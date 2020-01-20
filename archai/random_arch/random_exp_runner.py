from typing import Type

from ..common.config import Config
from ..nas import nas_utils
from ..nas.exp_runner import ExperimentRunner
from .random_micro_builder import RandomMicroBuilder
from ..nas.arch_trainer import ArchTrainer

from overrides import overrides

class RandomExperimentRunner(ExperimentRunner):
    @overrides
    def _run_search(self, conf_search:Config) -> None:
        # region config
        conf_model_desc = conf_search['model_desc']
        final_desc_filename = conf_search['final_desc_filename']
        # endregion

        # create model and save it to yaml
        # NOTE: there is no search here as the models are just randomly sampled
        model_desc = nas_utils.create_macro_desc(conf_model_desc,
                                                aux_tower=False,
                                                template_model_desc=None)
        macro_builder = self.micro_builder()
        macro_builder.build(model_desc, 0)

        # save model to location specified by eval config
        model_desc.save(final_desc_filename)

    @overrides
    def micro_builder(self)->RandomMicroBuilder:
        return RandomMicroBuilder()

    @overrides
    def trainer_class(self)->Type[ArchTrainer]:
        return ArchTrainer

