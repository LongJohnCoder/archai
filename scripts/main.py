import argparse
from typing import Dict, Type

from archai.nas.exp_runner import ExperimentRunner
from archai.darts.darts_exp_runner import DartsExperimentRunner

def main():
    runner_types:Dict[str, Type[ExperimentRunner]] = {
        'darts': DartsExperimentRunner,
    }

    parser = argparse.ArgumentParser(description='NAS E2E Runs')
    parser.add_argument('--algo', type=str, default='darts',
                        help='NAS algo to run')
    parser.add_argument('--toy', action='store_false', default=True,
                        help='Run in toy mode just to check for compile errors')
    parser.add_argument('--search', action='store_false', default=True,
                        help='Run search')
    parser.add_argument('--eval', action='store_false', default=True,
                        help='Run eval')
    parser.add_argument('--exp-prefix', type=str, default='throghaway',
                        help='Experiment prefix is used for directory names')
    args, extra_args = parser.parse_known_args()

    runner_type:Type[ExperimentRunner] = runner_types[args.algo]
    runner = runner_type(f'confs/{args.algo}_cifar.yaml', args.exp_prefix, args.toy)

    runner.run(search=args.search, eval=args.eval)


if __name__ == '__main__':
    main()
