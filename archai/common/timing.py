from functools import wraps
import gc
import timeit
from typing import Dict
from runstats import Statistics
import logging

_timings:Dict[str, Statistics] = {}

def add_timing(name:str, elapsed:float, no_print=True)->Statistics:
    global _timings
    stats = _timings.get(name, None)
    if stats is None:
        stats = Statistics()
        _timings[name] = stats
    stats.push(elapsed)

    if not no_print:
        logging.info('Timing "{}": {}s'.format(name, elapsed))
    return stats

def get_timing(name:str)->Statistics:
    return _timings.get(name)

def print_timing(name:str)->None:
    global _timings
    stats = _timings.get(name, None)
    if stats is None:
        logging.info(f'timing_name="{name}", avg=never_recorded')
    else:
        count = len(stats)
        logging.info(f'timing_name="{name}", '
              f'avg={stats.mean():.4g} '
              f'count={count} '
              f'stddev={stats.stddev() if count > 1 else float("NaN"):.4g} '
              f'min={stats.minimum():.4g} '
              f'max={stats.maximum():.4g} '
             )

def print_all_timings()->None:
    global _timings
    for name in _timings.keys():
        print_timing(name)

def clear_timings() -> None:
    global _timings
    _timings.clear()

def MeasureTime(f, no_print=True, disable_gc=False):
    @wraps(f)
    def _wrapper(*args, **kwargs):
        gcold = gc.isenabled()
        if disable_gc:
            gc.disable()
        start_time = timeit.default_timer()
        try:
            result = f(*args, **kwargs)
        finally:
            elapsed = timeit.default_timer() - start_time
            if disable_gc and gcold:
                gc.enable()
            name = f.__name__
            add_timing(name, elapsed, no_print=no_print)
        return result
    return _wrapper

class MeasureBlockTime:
    def __init__(self,name:str, no_print=True, disable_gc=False):
        self.name = name
        self.no_print = no_print
        self.disable_gc = disable_gc
    def __enter__(self):
        self.gcold = gc.isenabled()
        if self.disable_gc:
            gc.disable()
        self.start_time = timeit.default_timer()
    def __exit__(self,ty,val,tb):
        self.elapsed = timeit.default_timer() - self.start_time
        if self.disable_gc and self.gcold:
            gc.enable()
        add_timing(self.name, self.elapsed, no_print=self.no_print)
        return False #re-raise any exceptions
