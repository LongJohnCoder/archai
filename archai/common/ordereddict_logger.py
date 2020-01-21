from typing import Any, Optional
from collections import OrderedDict
import logging

import yaml


class OrderedDictLogger:
    def __init__(self, filepath:Optional[str], logger:Optional[logging.Logger], save_freq=1) -> None:
        super().__init__()
        self._logger = logger
        self._stack = [OrderedDict()]
        self._filepath = filepath
        self._save_freq = save_freq

    def log(self, key:Any, val:Any, level:int=logging.INFO)->None:
        if self._logger:
            msg = f'{key}'if val is None else f'{key}={val}'
            self._logger.log(level=level, msg=msg)

        self.insert(key, val)

    def root(self)->OrderedDict:
        return self._stack[0]

    def cur(self)->OrderedDict:
        return self._stack[-1]

    def save(self, filepath:str)->None:
        with open(filepath, 'w') as f:
            yaml.dump(self.root(), f)

    def load(self, filepath:str)->None:
        with open(filepath, 'r') as f:
            od = yaml.load(f, Loader=yaml.FullLoader)
            self._stack = [od]

    def insert(self, key:Any, val:Any):
        if key in self.cur():
            raise KeyError(f'Key "{key}" already exists in log')
        self.cur()[key] = val

    def begin(self, key:str)->'OrderedDictLogger':
        val = OrderedDict()
        self.insert(key, val)
        self._stack.append(val)
        return self

    def end(self):
        if len(self._stack)==1:
            raise RuntimeError('There is no child logger, end() call is invalid')
        self._stack.pop()

    def __enter__(self, key:str)->'OrderedDictLogger':
        return self.begin(key)
    def __exit__(self, type, value, traceback):
        self.end()

    def __contains__(self, key:Any):
        return key in self.cur()

    def __len__(self)->int:
        return len(self.cur())
