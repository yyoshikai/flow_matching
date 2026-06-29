from logging import Logger
from collections.abc import Container
from typing import overload
import numpy as np
import torch
import torch.nn as nn
from .train import Data, TSampler, Streamer, StopCriterion

# TSampler
class UniformTSampler(TSampler):
    def __init__(self, n_t_step: int):
        self.n_t_step = n_t_step
    def __call__(self):
        return float(np.random.randint(0, self.n_t_step) / self.n_t_step)


# Range
class Range(Container):
    def __contains__(self, x: int):
        raise NotImplementedError

class CatRange(Range):
    def __init__(self, *ranges: Range):
        self.ranges = ranges
    def __contains__(self, x: int):
        return any(x in r for r in self.ranges)

class GERange(Range):
    def __init__(self, n: int):
        super().__init__()
        self.n = n
    def __contains__(self, x: int):
        return x >= self.n

class RepeatRange(Range):
    @overload
    def __init__(self, step: int): ...
    @overload
    def __init__(self, start: int, step: int): ...
    def __init__(self, a: int, b: int|None=None):
        if b is None:
            self.start, self.step = a, a
        else:
            self.start, self.step = a, b
    def __contains__(self, x: int):
        if x >= self.start and (x-self.start) % self.step == 0:
            return True
        return False

class AmpRange(Range):
    def __init__(self, min_step: int=1, max_step: int|None=None):
        self.min_step = min_step
        self.max_step = max_step
    def __init__(self, x):
        if x >= self.max_step:
            return x % self.max_step == 0
        else:
            if x % self.min_step != 0:
                return False
            n = x // self.min_step
            if (n&(n-1)) == 0:
                return True
            else:
                return False


# Streamer
class Streamers(Streamer):
    def __init__(self, streamers: list[Streamer]):
        self.streamers = streamers
    def put(self, model: nn.Module, batch_data: list[Data]):
        for streamer in self.streamers:
            streamer.put(model, batch_data)

class SaveModelStreamer(Streamer):
    def __init__(self, path_format: str, range: Container):
        self.path_format = path_format
        self.range = range
        self.step = 0
    def put(self, model: nn.Module, batch_data: list[Data]):
        self.step += 1
        if self.step in self.range:
            path = self.path_format.format(step=self.step)
            torch.save(model.state_dict(), path)

class LogStepStreamer(Streamer):
    def __init__(self, logger: Logger, range: Container):
        self.logger = logger
        self.range = range
        self.step = 0
    def put(self, model: nn.Module, batch_data: list[Data]):
        self.step += 1
        if self.step in self.range:
            self.logger.debug(f"Finished step={self.step}")


# StopCriterion
class AnyStopCriterion(StopCriterion):
    def __init__(self, criteria: list[StopCriterion]):
        self.criteria = criteria
    def __call__(self, model, batch_data):
        return any(criterion(model, batch_data) for criterion in self.criteria)

class StepStopCriterion(StopCriterion):
    def __init__(self, step: int):
        self.max_step = step
        self.cur_step = 0
    def __call__(self, model, batch_data):
        self.cur_step += 1
        return self.max_step <= self.cur_step
