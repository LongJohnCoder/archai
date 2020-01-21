import argparse
from typing import Dict, Type

from archai.nas.exp_runner import ExperimentRunner
from archai.darts.darts_exp_runner import DartsExperimentRunner
from archai.petridish.petridish_exp_runner import PetridishExperimentRunner
from archai.random_arch.random_exp_runner import RandomExperimentRunner

def main():
    runner_types:Dict[str, Type[ExperimentRunner]] = {
        'darts': DartsExperimentRunner,
        'petridish': PetridishExperimentRunner,
        'random': RandomExperimentRunner
    }

    parser = argparse.ArgumentParser(description='NAS E2E Runs')
    parser.add_argument('--algos', type=str, default='darts,petridish,random',
                        help='NAS algos to run seprated by comma')
    parser.add_argument('--full', action='store_true', default=False,
                        help='Run in full or toy mode just to check for compile errors')
    parser.add_argument('--no-search', action='store_true', default=False,
                        help='Run search')
    parser.add_argument('--no-eval', action='store_true', default=False,
                        help='Run eval')
    parser.add_argument('--exp-prefix', type=str, default='throwaway',
                        help='Experiment prefix is used for directory names')
    args, extra_args = parser.parse_known_args()

    for algo in args.algos.split(','):
        algo = algo.strip()
        print('Running: ', algo)
        runner_type:Type[ExperimentRunner] = runner_types[algo]
        runner = runner_type(f'confs/{algo}_cifar.yaml',
                             base_name=f'{algo}_{args.exp_prefix}',
                             toy=not args.full)

        runner.run(search=not args.no_search, eval=not args.no_eval)


if __name__ == '__main__':
    main()
