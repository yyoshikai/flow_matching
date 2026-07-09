import os, random
import itertools as itr
from logging import Logger
from collections.abc import Container
from typing import overload
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler
from .train import Streamer, StopCriterion

# Range
class Range(Container):
    def __contains__(self, x: int):
        raise NotImplementedError

class CatRange(Range):
    def __init__(self, *ranges: Container):
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
    def __contains__(self, x):
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
class Streamers(list[Streamer], Streamer):
    def __init__(self, streamers: list[Streamer]):
        super().__init__(streamers)
    def put_data(self, batch):
        for streamer in self:
            streamer.put_data(batch)
    def put_loss(self, model, loss):
        for streamer in self:
            streamer.put_loss(model, loss)
    def put_optim(self, model, optimizer):
        for streamer in self:
            streamer.put_optim(model, optimizer)

class SaveModelStreamer(Streamer):
    def __init__(self, path_format: str, range: Container):
        self.path_format = path_format
        self.range = range
        self.step = 0
    def put_optim(self, model, optimizer):
        self.step += 1
        if self.step in self.range:
            path = self.path_format.format(step=self.step)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(model.state_dict(), path)

class LogStepStreamer(Streamer):
    def __init__(self, logger: Logger, range: Container):
        self.logger = logger
        self.range = range
        self.step = 0
    def put_optim(self, model, optimizer):
        self.step += 1
        if self.step in self.range:
            self.logger.debug(f"Finished step={self.step}")

class SaveLossStreamer(Streamer):
    def __init__(self, path: str):
        self.path = path
        self.step = 0
        self.loss_names = None
    def put_loss(self, model, loss):
        if self.step == 0:
            self.loss_names = loss.names
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, 'w') as f:
                f.write(','.join(['step']+self.loss_names)+'\n')
        else:
            assert self.loss_names == loss.names
        with open(self.path, 'a') as f:
            row = [self.step]+[l.item() for l in loss.losses]
            f.write(','.join(map(str, row))+'\n')
        self.step += 1

class SaveGradStreamer(Streamer):
    def __init__(self, path_format: str, range: Container):
        self.path_format = path_format
        self.range = range
        self.step = 0
    def put_loss(self, model, loss):
        self.step += 1
        if self.step not in self.range:
            return
        names, params = zip(*[(name, p) for name, p in model.named_parameters() if p.requires_grad])
        for k, l in zip(loss.names, loss.losses):
            grads = torch.autograd.grad(
                outputs=l, inputs=params, retain_graph=True, allow_unused=True
            )
            grad_state = {name: grad for name, grad in zip(names, grads)}
            path = self.path_format.format(step=str(self.step), k=k)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(grad_state, path)
        
        # model weight
        path = self.path_format.format(step=str(self.step), k='weight')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model.state_dict(), path)


class Optimizer:
    def __init__(self, optimizer: torch.optim.Optimizer, scheduler: LRScheduler|None, clip_grad_norm: float|None):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.clip_grad_norm = clip_grad_norm
    def step(self):
        if self.clip_grad_norm is not None:
            params = itr.chain(*[group['params'] for group in self.optimizer.param_groups])
            nn.utils.clip_grad_norm_(params, self.clip_grad_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
    def zero_grad(self):
        self.optimizer.zero_grad()


# StopCriterion
class AnyStopCriterion(StopCriterion):
    def __init__(self, criteria: list[StopCriterion]):
        self.criteria = criteria
    def __call__(self, model, batch_data, loss):
        return any(criterion(model, batch_data, loss) for criterion in self.criteria)

class StepStopCriterion(StopCriterion):
    def __init__(self, step: int):
        self.max_step = step
        self.cur_step = 0
    def __call__(self, model, batch_data, loss):
        self.cur_step += 1
        return self.max_step <= self.cur_step
