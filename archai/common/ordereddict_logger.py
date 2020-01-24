from typing import Any, Mapping, Optional, Union, List
from collections import OrderedDict
import logging
import time
import atexit
from functools import partial
import itertools
import yaml

TItems = Union[Mapping, str]

class OrderedDictLogger:
    def __init__(self, filepath:Optional[str], logger:Optional[logging.Logger],
                 save_delay:Optional[float]=1.0) -> None:
        super().__init__()
        self._logger = logger
        self._stack:List[Optional[OrderedDict]] = [OrderedDict()]
        self._path = [['']]
        self._filepath = filepath
        self._save_delay = save_delay
        self._call_count = 0
        self._last_save = time.time()

        atexit.register(self.save)

    def warn(self, dict:TItems, level:Optional[int]=logging.WARN, exists_ok=False)->None:
        self.info(dict, level, exists_ok)

    def info(self, dict:TItems, level:Optional[int]=logging.INFO, exists_ok=False)->None:
        self._call_count += 1

        if isinstance(dict, Mapping):
            self._update(dict, exists_ok)
            msg = ', '.join(f'{k}={v}' for k, v in dict.items())
        else:
            msg = dict
            key = '_warnings' if level==logging.WARN else '_messages'
            self._update_key(self._call_count, msg, node=self._root(), path=[key])

        if level is not None and self._logger:
            self._logger.log(msg='|'.join(self.path()) + ' ' + msg, level=level)

        if self._save_delay is not None and \
                time.time() - self._last_save > self._save_delay:
            self.save()
            self._last_save = time.time()

    def _root(self)->OrderedDict:
        r = self._stack[0]
        assert r is not None
        return r

    def _cur(self)->OrderedDict:
        self._ensure_path()
        c = self._stack[-1]
        assert c is not None
        return c

    def save(self, filepath:Optional[str]=None)->None:
        filepath = filepath or self._filepath
        if filepath:
            with open(filepath, 'w') as f:
                yaml.dump(self._root(), f)

    def load(self, filepath:str)->None:
        with open(filepath, 'r') as f:
            od = yaml.load(f, Loader=yaml.FullLoader)
            self._stack = [od]

    def _insert(self, dict:Mapping):
        self._update(dict, exists_ok=False)

    def _update(self, dict:Mapping, exists_ok=True):
        for k,v in dict.items():
            self._update_key(k, v, exists_ok)

    def _update_key(self, key:Any, val:Any, exists_ok=True,
                    node:Optional[OrderedDict]=None, path:List[str]=[]):
        if not exists_ok and key in self._cur():
            raise KeyError(f'Key "{key}" already exists in log')

        node = node if node is not None else self._cur()
        for p in path:
            if p not in node:
                node[p] = OrderedDict()
            node = node[p]
        node[str(key)] = val

    def _ensure_path(self)->None:
        if self._stack[-1] is not None:
            return
        last_od = None
        for i, (path, od) in enumerate(zip(self._path, self._stack)):
            if od is None:
                od = last_od
                for key in path:
                    if key not in od:
                        od[key] = OrderedDict()
                    assert isinstance(od[key], OrderedDict)
                    od = od[key]
                self._stack[i] = od
            last_od = od

    def begin(self, *keys:Any)->'OrderedDictLogger':
        self._path.append([str(k) for k in keys])
        self._stack.append(None) # delay create

        return self # this allows calling __enter__

    def end(self):
        if len(self._stack)==1:
            raise RuntimeError('There is no child logger, end() call is invalid')
        self._stack.pop()
        self._path.pop()

    def path(self)->List[str]:
        # flatten array of array
        return list(itertools.chain.from_iterable(self._path))
    def __enter__(self)->'OrderedDictLogger':
        return self
    def __exit__(self, type, value, traceback):
        self.end()

    def __contains__(self, key:Any):
        return key in self._cur()

    def __len__(self)->int:
        return len(self._cur())
