import math
import itertools as itr
from dataclasses import dataclass
from collections.abc import Iterator
from typing import Self, Any
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader

class Path[D, Tgt, BPred]:
    def sample(self, data0: D, data1: D, t: float) -> tuple[D, Tgt]:
        raise NotImplementedError
    
    def update(self, datas: list[D], bpred: BPred, t: float, delta_t: float) -> list[D]:
        raise NotImplementedError

    def criterion(self, targets: list[Tgt], bpred: BPred) -> Loss:
        raise NotImplementedError

class FMModel[D, BPred](nn.Module):
    def forward(self, datas: list[D], ts: list[float]) -> BPred:
        raise NotImplementedError

@dataclass
class Loss:
    losses: list[Tensor]
    names: list[str]
    weights: list[float]

    def loss(self):
        return sum(loss*weight for loss, weight in zip(self.losses, self.weights))
    
    @classmethod
    def cat(cls, *cinfos: Loss):
        losses = list(itr.chain(*[cinfo.losses for cinfo in cinfos]))
        names = list(itr.chain(*[cinfo.names for cinfo in cinfos]))
        weights = list(itr.chain(*[cinfo.weights for cinfo in cinfos]))
        return Loss(losses, names, weights)

class Streamer[D, Tgt]:
    def put_data(self, batch: list[tuple[D, float, Tgt]]):
        pass
    def put_loss(self, model: nn.Module, loss: Loss) -> None:
        pass
    def put_optim(self, model: nn.Module, optimizer: Optimizer):
        pass

class StopCriterion[D]:
    def __call__(self, model: nn.Module, batch_data: list[D], loss: Tensor) -> bool:
        raise NotImplemented

def train_fm[D, Tgt, BPred](
    fm_model: FMModel[D], 
    optimizer: Optimizer,
    data_iter: Iterator[tuple[D, float, Tgt]],
    path: Path[D, Tgt, BPred],
    streamer: Streamer,
    stop_criterion: StopCriterion,
):
    fm_model.train()
    while True:
        batch_data = data_iter.__next__()
        streamer.put_data(batch_data)
        datas, ts, targets = zip(*batch_data)
        optimizer.zero_grad()
        bpred = fm_model(datas, ts)
        loss = path.criterion(targets, bpred)
        streamer.put_loss(fm_model, loss)
        loss.loss().backward()
        optimizer.step()
        streamer.put_optim(fm_model, optimizer)
        if stop_criterion(fm_model, batch_data, loss):
            break

class Distribution[D]:
    def sample(self) -> D:
        raise NotImplementedError

    def bsample(self, n: int) -> list[D]:
        return [self.sample() for i in range(n)]
