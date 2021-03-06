
import logging

from archai.common import data
from archai.common import utils
from archai.common.timing import MeasureTime, print_all_timings, print_timing, get_timing
from archai.common.common import get_logger, common_init


conf = common_init(config_filepath='confs/darts_cifar.yaml',
                    param_args=['--common.experiment_name', 'restnet_test'])

conf_eval = conf['nas']['eval']
conf_loader       = conf_eval['loader']
conf_loader['train_batch'] = 512
conf_loader['test_batch'] = 4096
conf_loader['cutout'] = 0
train_dl, _, test_dl = data.get_data(conf_loader)


@MeasureTime
def iter_dl(dl):
    dummy = 0.0
    for x,y in train_dl:
        x = x.cuda()
        y = y.cuda()
        dummy += len(x)
       # dummy += len(y)
    return dummy

logging.info(f'batch_cout={len(train_dl)}')

dummy = 0.0
for _ in range(10):
    dummy = iter_dl(train_dl)
    print(dummy)

print_all_timings()
logging.shutdown()
exit(0)