import torch.nn as nn
from collections.abc import Iterator
from typing import Self
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader

class Data:
    def update(self, vec: 'Vec[Self]', delta_t: float):
        raise NotImplementedError

class Vec[DataT]:
    pass

class VecList[VecT]:
    def to_list(self) -> list[VecT]:
        raise NotImplementedError

class FMModel[DataT](nn.Module):
    def forward(self, datas: list[DataT], ts: list[float]) -> VecList[Vec[DataT]]:
        raise NotImplementedError
    
class TSampler:
    def __call__(self) -> float:
        raise NotImplementedError
    def iter_t_step(self) -> Iterator[float, float]:
        raise NotImplementedError

class PSampler[DataT]:
    def sample(self, data1: DataT, t: float) -> tuple[DataT, Vec[DataT]]:
        raise NotImplementedError
    def sample_from_0(self) -> DataT:
        raise NotImplementedError
    
class Loss:
    def backward(self):
        raise NotImplementedError

class Criterion[DataT, LossT](nn.Module):
    def forward(self, vecs_true: list[Vec[DataT]], vecs_pred: VecList[Vec[DataT]]) -> LossT:
        raise NotImplementedError

class Streamer:
    def put_data(self, batch: list[Data]):
        pass
    def put_loss(self, model: nn.Module, loss: Tensor|Loss) -> None:
        pass
    def put_optim(self, model: nn.Module):
        pass

class StopCriterion:
    def __call__(self, model: nn.Module, batch_data: list[Data], loss: Tensor|Loss) -> bool:
        raise NotImplemented

class TrainFMDataset[DataT](Dataset[tuple[DataT, float, Vec[DataT]]]):
    def __init__(self, dataset: Dataset[DataT], t_sampler: TSampler, p_sampler: PSampler[DataT]):
        self.dataset = dataset
        self.t_sampler = t_sampler
        self.p_sampler = p_sampler
    
    def __getitem__(self, idx):
        data1 = self.dataset[idx]
        t = self.t_sampler.sample()
        data, vec = self.p_sampler.sample(data1, t)
        return data, t, vec

def train_fm[DataT](
    fm_model: FMModel[DataT], 
    optimizer: Optimizer,
    data_iter: Iterator[tuple[DataT, float, Vec[DataT]]],
    criterion: Criterion[DataT],
    streamer: Streamer,
    stop_criterion: StopCriterion,
):
    fm_model.train()
    while True:
        batch_data = data_iter.__next__()
        streamer.put_data(batch_data)
        datas, ts, vecs = zip(*batch_data)
        vecs_out: VecList[DataT] = fm_model(datas, ts)
        loss = criterion(vecs, vecs_out)
        streamer.put_loss(fm_model, loss)
        loss.backward()
        optimizer.step()
        streamer.put_optim(fm_model)
        if stop_criterion(fm_model, batch_data, loss):
            break









