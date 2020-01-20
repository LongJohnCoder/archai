from typing import Callable, Tuple, Optional

from torch import nn, Tensor
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from overrides import EnforceOverrides

from .metrics import Metrics
from .tester import Tester
from .config import Config
from . import utils
from ..common.common import get_logger
from ..common.check_point import CheckPoint
from .apex_utils import Amp

class Trainer(EnforceOverrides):
    def __init__(self, conf_train:Config, model:nn.Module, device,
                 check_point:Optional[CheckPoint], aux_tower:bool)->None:
        logger = get_logger()

        # region config vars
        conf_lossfn = conf_train['lossfn']
        self._apex = conf_train['apex']
        self._aux_tower = aux_tower
        self._aux_weight = conf_train['aux_weight']
        self._grad_clip = conf_train['grad_clip']
        self._drop_path_prob = conf_train['drop_path_prob']
        self._logger_freq = conf_train['logger_freq']
        self._title = conf_train['title']
        self._epochs = conf_train['epochs']
        self._conf_optim = conf_train['optimizer']
        self._conf_sched = conf_train['lr_schedule']
        conf_validation = conf_train['validation']
        self._validation_freq = 0 if conf_validation is None else conf_validation['freq']
        # endregion

        self.check_point = check_point
        self.model = model
        self.device = device
        self._lossfn = utils.get_lossfn(conf_lossfn).to(device)
        self._tester = Tester(conf_validation, model, device,
                              aux_tower=aux_tower, epochs=self._epochs) \
                        if conf_validation else None
        self._metrics = self._create_metrics(self._epochs)
        self._metrics.custom['param_byte_size'] = utils.param_size(self.model)
        logger.info("Model param size = %f MB", self._metrics.custom['param_byte_size']/1e6)

        self._amp = Amp(self._apex)

    def fit(self, train_dl:DataLoader, val_dl:Optional[DataLoader])->None:
        logger = get_logger()
        # optimizers, schedulers needs to be recreated for each fit call
        # as they have state
        optim = self.create_optimizer()
        # apply scheduler before amp
        self._sched, self._sched_on_epoch = self._create_scheduler(optim, len(train_dl))
        # before checkpoint restore, convert to amp
        # TODO: original model is lost after to_amp?
        self.model, self._optim = self._amp.to_amp(self.model, optim)

        start_epoch = 0
        if self.check_point is not None and 'trainer' in self.check_point:
            start_epoch = self._restore_checkpoint()

        self.pre_fit(train_dl, val_dl, start_epoch>0)

        if start_epoch >= self._epochs:
            logger.warn(f'fit exit: start epoch {start_epoch} > {self._epochs}')
            return # we already finished the run, we might be checkpointed

        if self.check_point is not None:
            self.check_point.clear()
        for epoch in range(start_epoch, self._epochs):
            self._set_drop_path(epoch, self._epochs)

            self.pre_epoch(train_dl, val_dl)
            self._train_epoch(train_dl)
            self.post_epoch(train_dl, val_dl)

        self.post_fit(train_dl, val_dl)

        # make sure we don't keep references to the graph
        del self._optim
        del self._sched

    def create_optimizer(self)->Optimizer:
        return utils.create_optimizer(self._conf_optim, self.model.parameters())

    def _create_scheduler(self, optim:Optimizer, steps_per_epoch:int) \
            ->Tuple[Optional[_LRScheduler],bool]:
        return utils.create_lr_scheduler(self._conf_sched, self._epochs,
            optim, steps_per_epoch)

    def get_optimizer(self)->Optimizer:
        return self._optim
    def get_scheduler(self)->Optional[_LRScheduler]:
        return self._sched

    def get_metrics(self)->Tuple[Metrics, Optional[Metrics]]:
        return self._metrics, self._tester.get_metrics() if self._tester else None

    #########################  hooks #########################
    def pre_fit(self, train_dl:DataLoader, val_dl:Optional[DataLoader],
                resuming:bool)->None:
        self._metrics.pre_run(resuming)
        if self._tester:
            self._tester.pre_test(resuming)

    def post_fit(self, train_dl:DataLoader, val_dl:Optional[DataLoader])->None:
        if self._tester:
            self._tester.post_test()
        self._metrics.post_run()

    def pre_epoch(self, train_dl:DataLoader, val_dl:Optional[DataLoader])->None:
        self._metrics.pre_epoch(lr=self._optim.param_groups[0]['lr'])

    def post_epoch(self, train_dl:DataLoader, val_dl:Optional[DataLoader])->None:
        # first run test before checkpointing
        if val_dl and self._tester and self._validation_freq > 0:
            if self._metrics.epoch+1 % self._validation_freq == 0 or \
                    self._metrics.epoch+1 >= self._epochs:
                self._tester.test_epoch(val_dl)
            else:
                self._tester.increment_epoch()

        self._metrics.post_epoch()
        if self.check_point is not None and \
                self._metrics.epoch % self.check_point.freq == 0:
            self.check_point.new()
            self.update_checkpoint(self.check_point)
            self.check_point.commit()

    def pre_step(self, x:Tensor, y:Tensor)->None:
        self._metrics.pre_step(x, y)

    def post_step(self, x:Tensor, y:Tensor, logits:Tensor, loss:Tensor,
                  steps:int)->None:
        self._metrics.post_step(x, y, logits, loss, steps)
    #########################  hooks #########################

    def _restore_checkpoint(self)->int:
        logger = get_logger()

        last_epoch = self._metrics.epoch
        assert last_epoch > 0, f'While restoring from checkpoint epoch > 0 is expected but it is {last_epoch}'
        start_epoch = last_epoch + 1

        state = self.check_point['trainer']
        self._amp.load_state_dict(state['amp'])
        self.model.load_state_dict(state['model'])
        self._optim.load_state_dict(state['optim'])
        if self._sched:
            self._sched.load_state_dict(state['sched'])
        else:
            assert state['sched'] is None
        if self._tester:
            self._tester.load_state_dict(state['tester'])

        logger.warn(f'fit will continue from epoch {start_epoch}')
        return start_epoch

    def update_checkpoint(self, check_point:CheckPoint)->None:
        state = {
            'metrics': self._metrics.state_dict(),
            'model': self.model.state_dict(),
            'optim': self._optim.state_dict(),
            'sched': self._sched.state_dict() if self._sched else None,
            'amp': self._amp.state_dict(),
            'tester': self._tester.state_dict() if self._tester is not None else None
        }
        self.check_point['trainer'] = state

    def _create_metrics(self, epochs:int):
        logger = get_logger()
        m = Metrics(self._title, epochs,logger_freq=self._logger_freq)
        if self.check_point is not None and 'trainer' in self.check_point:
            logger.warn('Metrics loaded from exisitng checkpoint')
            m.load_state_dict(self.check_point['trainer']['metrics'])
        return m

    def _train_epoch(self, train_dl: DataLoader)->None:
        steps = len(train_dl)
        self.model.train()
        # TODO: advantage of doing at the start is that if schedule starts from 0
        #       first epoch is not a waste. But then again, we lose first LR.
        if self._sched and self._sched_on_epoch:
            self._sched.step()
        for x, y in train_dl:
            assert self.model.training # derived class might alter the mode

            # enable non-blocking on 2nd part so its ready when we get to it
            x, y = x.to(self.device), y.to(self.device, non_blocking=True)

            self.pre_step(x, y)

            self._optim.zero_grad()

            logits, aux_logits = self.model(x), None
            if self._aux_tower:
                assert isinstance(logits, Tuple) and len(logits) >=2
                logits, aux_logits = logits[0], logits[1]
            loss = self.compute_loss(self._lossfn, x, y, logits,
                                    self._aux_weight, aux_logits)

            self._amp.backward(loss, self._optim)

            # TODO: original darts clips alphas as well but pt.darts doesn't
            self._amp.clip_grad(self._grad_clip, self.model, self._optim)

            self._optim.step()
            if self._sched and not self._sched_on_epoch:
                self._sched.step()

            self.post_step(x, y, logits, loss, steps)

    def compute_loss(self, lossfn:Callable,
                     x:Tensor, y:Tensor, logits:Tensor,
                     aux_weight:float, aux_logits:Optional[Tensor])->Tensor:
        loss = lossfn(logits, y)
        if aux_weight > 0.0 and  aux_logits is not None:
            loss += aux_weight * lossfn(aux_logits, y)
        return loss

    def _set_drop_path(self, epoch:int, epochs:int)->None:
        if self._drop_path_prob:
            drop_prob = self._drop_path_prob * epoch / epochs
            # set value as property in model (it will be used by forward())
            # this is necessory when using DataParallel(model)
            # https://github.com/pytorch/pytorch/issues/16885
            m = self.model
            if hasattr(self.model, 'module'): # for data parallel model
                m = self.model.module
            if hasattr(m, 'drop_path_prob'):
                m.drop_path_prob(drop_prob)
            else:
                raise RuntimeError('Drop path value {} was specified but model'
                                   ' does not have drop_path_prob() method'\
                                       .format(self._drop_path_prob))
