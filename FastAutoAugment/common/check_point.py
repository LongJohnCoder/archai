from collections import UserDict
from typing import Callable, Any
import weakref
import os

import torch

from .config import Config
from .common import logdir_abspath

_CallbackType = Callable #[['CheckPoint', *kargs: Any, **kwargs: Any], None]
class CheckPoint(UserDict):
    """Callback based checkpoint model.

    Start new checkpoint by calling new() and save it by calling save().
    This class is also dictionaty. Items that needs be saved can be done so
    by setting key, value pairs after new(). As any dictionary key is set,
    checkpoint becomes dirty. On call to new() subscribers will be notified so
    they can insert their items in the dictionary.
    Invariant: checkpoint is dirty only between new() and save() calls.
    """
    def __init__(self, conf_checkpoint:Config, load_existing:bool) -> None:
        super().__init__()

        # region config vars
        self._filepath = logdir_abspath(conf_checkpoint['filename'])
        self.frequency = conf_checkpoint['frequency']
        # endregion

        self._callbacks = []

        if load_existing:
            self.load_existing()

    def load_existing(self)->bool:
        assert self.is_empty()
        if self._filepath and os.path.exists(self._filepath):
            d = torch.load(self._filepath,
                           map_location=lambda storage, loc: storage)
            self.clear()
            self.update(d)
            return True
        return False

    def new(self, *kargs, **kvargs)->None:
        assert self.is_empty(), 'checkpoint is dirty so new cannot be started'
        for func, obj in self._callbacks:
            func = func() # get actual refrence from weakref
            if obj is not None:
                obj = obj() # get actual reference from weakref
                if obj is None:
                    continue # instance is gone
                func(obj, self, *kargs, **kvargs)
            elif func is not None:
                func(self, *kargs, **kvargs)
            # else func is garbegde collected

    def commit(self)->None:
        assert self._filepath and not self.is_empty()
        torch.save(self.data, self._filepath)
        # clean up after commit so we don't hold up references
        self.clear()

    def is_empty(self)->bool:
        return len(self) == 0

    def subscribe(self, callback:_CallbackType)->None:
        obj = getattr(callback, '__self__', None)
        callback_ref = weakref.ref(callback.__func__), \
                       None if obj is None else weakref.ref(obj)
        self._callbacks.append(callback_ref)