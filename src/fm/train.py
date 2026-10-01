import math
import itertools as itr
from dataclasses import dataclass
from collections.abc import Iterator, Callable
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer

class Path[D, Tgt, BPred]:
    def sample(self, data0: D, data1: D, t: float) -> tuple[D, Tgt]:
        raise NotImplementedError
    def update(self, datas: list[D], bpred: BPred, t0: float, t1: float) -> list[D]:
        raise NotImplementedError
    def criterion(self, targets: list[Tgt], bpred: BPred) -> Loss:
        raise NotImplementedError

class FMModel[D, BPred](nn.Module):
    def forward(self, datas: list[D], ts: list[int]) -> BPred:
        raise NotImplementedError

@dataclass
class Loss:
    losses: list[Tensor]
    names: list[str]
    weights: list[float]

    def loss(self):
        return sum(loss*weight for loss, weight in zip(self.losses, self.weights))
    
    @classmethod
    def cat(cls, cinfos: list[Loss], cnames: list[str], cweights: list[float]):
        losses = []
        names = []
        weights = []
        for i, cinfo in enumerate(cinfos):
            losses += cinfo.losses
            names += [cnames[i]+'_'+name for name in cinfo.names]
            weights += [w*cweights[i] for w in cinfo.weights]
        return Loss(losses, names, weights)

class Streamer[D, Tgt]:
    def start(self, model: nn.Module):
        pass
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
    path: Path[Tgt, BPred],
    streamer: Streamer,
    stop_criterion: StopCriterion,
):
    fm_model.train()
    streamer.start(fm_model)
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

class GStreamer[D, BPred]:
    def init(self, data: D, batch_idx: int):
        pass
    def put(self, data: D, bpred: BPred, t: float, delta_t: float):
        pass
    def end(self):
        pass

@torch.inference_mode()
def generate[D, Tgt, BPred](
    fm_model: FMModel[D],
    path: Path[D, Tgt, BPred],
    gstreamers: list[GStreamer[D, BPred]],
    ts: list[tuple[float, float]],
    max_batch_size: int,
    n: int
):
    
    all_datas = []
    for idxs in itr.batched(range(n), max_batch_size):
        datas = path.bsample_init(len(idxs))
        for bidx, idx in enumerate(idxs):
            gstreamers[idx].init(datas[bidx], bidx)
        for t, delta_t, alpha in ts:
            bpred = fm_model(datas, [t]*len(idxs))
            datas = path.update(datas, bpred, t, delta_t, alpha)
            for idx, data in zip(idxs, datas):
                gstreamers[idx].put(data, bpred, t, delta_t)
        for idx in idxs:
            gstreamers[idx].end()
        all_datas += datas
    return all_datas