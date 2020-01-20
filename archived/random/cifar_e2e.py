import subprocess
import argparse
import os
from pathlib import Path
import shutil

from archai.common.common import common_init, get_conf_common
from archai.common import utils

def main():
    # accept search and eval scripts to run
    # config file can be supplied using --config
    parser = argparse.ArgumentParser(description='NAS E2E Runs')
    parser.add_argument('--search-script', type=str,
                        default='scripts/random/cifar_search.py',
                        help='Search script to run')
    parser.add_argument('--eval-script', type=str,
                        default='scripts/random/cifar_eval.py',
                        help='Eval script to run')
    parser.add_argument('--exp_prefix', type=str, default='random',
                        help='Experiment prefix to use')
    args, extra_args = parser.parse_known_args()

    # load config to some of the settings like logdir
    conf = common_init(use_args=True)
    logdir = get_conf_common()['logdir']
    assert logdir

    # get script, resume flag and experiment dir for search
    search_script = args.search_script
    resume = conf['nas']['search']['resume']
    search_script = utils.full_path(search_script.strip())
    experiment_name = args.exp_prefix + '_' + Path(search_script).stem
    experiment_dir = os.path.join(logdir, experiment_name)

    # see if search has already produced the output
    final_desc_filepath = os.path.join(experiment_dir, conf['nas']['search']['final_desc_filename'])
    if not resume or not os.path.exists(final_desc_filepath):
        print(f'Starting {search_script}...')
        result = subprocess.run(
            ['python', search_script,
            '--config', conf.config_filepath,
            '--config-defaults', conf.config_defaults_filepath,
            '--common.experiment_name', experiment_name
            ])
        print(f'Script {search_script} returned {result.returncode}')
        if result.returncode != 0:
            exit(result.returncode)
    else:
        print(f'Search is skipped because file {final_desc_filepath} already exists')

    # get script, resume flag and experiment dir for eval
    eval_script = args.eval_script
    resume = conf['nas']['eval']['resume']
    eval_script = utils.full_path(eval_script.strip())
    experiment_name = args.exp_prefix + '_' + Path(eval_script).stem
    experiment_dir = os.path.join(logdir, experiment_name)

    # if eval has already produced the output, skip eval run
    model_filepath = os.path.join(experiment_dir, conf['nas']['eval']['save_filename'])
    if not resume or not os.path.exists(model_filepath):
        # copy output of search to eval folder
        # TODO: take final_desc_filename from eval config
        os.makedirs(experiment_dir, exist_ok=True)
        shutil.copy2(final_desc_filepath, experiment_dir)

        print(f'Starting {eval_script}...')
        result = subprocess.run(
            ['python', eval_script,
            '--config', conf.config_filepath,
            '--config-defaults', conf.config_defaults_filepath,
            '--common.experiment_name', experiment_name
            ])
        print(f'Script {eval_script} returned {result.returncode}')
        if result.returncode != 0:
            exit(result.returncode)
    else:
        print(f'Eval is skipped because file {model_filepath} already exists')
    print('Search and eval done.')
    exit(0)

if __name__ == '__main__':
    try:
        main()
    except (KeyboardInterrupt):
        print('Interupted by keyboard')
    exit(0)