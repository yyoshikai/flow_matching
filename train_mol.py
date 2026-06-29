import multiprocessing
from dataclasses import dataclass
from typing import Literal
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import Tensor

from rdkit import Chem
from src.utils.logger import get_logger, add_file_handler
from src.data.datasets.unimol import UniMolLigandDataset
from src.data.sampler import InfiniteRandomSampler
from src.fm.train import Vec, TSampler, train_fm
from src.fm.utils import UniformTSampler, Streamers, SaveModelStreamer, LogStepStreamer, AmpRange, RepeatRange, StepStopCriterion
from src.mol.model import MolData, MolVec, GraphFMModel, MolVecCriterion
from src.mol.data import ATOMS

def get_random_rotation_matrix(rng: np.random.Generator):
    # get axes
    axes = []
    while(len(axes) < 2):
        new_axis = rng.random(3)
        
        new_norm = np.sqrt(np.sum(new_axis**2))
        if (new_norm < 0.1 or 1 <= new_norm): continue
        new_axis = new_axis / new_norm
        if np.any([np.abs(np.sum(axis*new_axis)) >= 0.9 for axis in axes]):
            continue
        axes.append(new_axis)

    # get rotation matrix
    axis0, axis1b = axes
    axis1 = np.cross(axis0, axis1b)
    axis1 = axis1 / np.linalg.norm(axis1)
    axis2 = np.cross(axis0, axis1)
    axis2 = axis2 / np.linalg.norm(axis2)
    return np.array([axis0, axis1, axis2])

class MolDater(Dataset[tuple[MolData, float, Vec[MolData]]]):
    def __init__(self):
        self.n_atom = 100
        self.init_coord_std = 100
        self.end_coord_std = 100

        self.no_atom_idx = 0
        self.atom2idx = {atom: i+1 for i, atom in enumerate(ATOMS)}
        self.mask_atom_idx = len(ATOMS)+1
        self.n_idx = len(ATOMS)+2
        self.rng = np.random.default_rng(0)
        pass

    def to_data(self, mol: Chem.Mol):
        n_no_atom = self.n_atom - mol.GetNumAtoms()
        if n_no_atom < 0:
            raise ValueError(f"{mol.GetNumAtoms()=}")
        node = torch.tensor([self.atom2idx[atom.GetSymbol()] for atom in mol.GetAtoms()]+[self.no_atom_idx]*n_no_atom, dtype=torch.long)
        
        coord = mol.GetConformer().GetPositions() # [Na, 3]
        coord = coord - np.mean(coord, axis=0)
        coord = np.matmul(coord, get_random_rotation_matrix(self.rng))

        coord = np.concatenate([
            mol.GetConformer().GetPositions(),
            self.rng.normal(size=(n_no_atom, 3))*self.init_coord_std
        ])
        return MolData(node, coord)

    def sample_init_data(self):
        node = torch.full((self.n_atom,), fill_value=self.mask_atom_idx, dtype=torch.long)
        coord = self.rng.normal(size=(self.n_atom, 3)) * self.init_coord_std
        return MolData(node, coord)

    def interleave(self, data0: MolData, data1: MolData, t) -> tuple[MolData, MolVec]:
        # node
        r = torch.rand((self.n_atom,))
        is_0 = r < (1-t)**2
        is_1 = r >= 1-t**2
        is_r = (~is_0)&(~is_1)
        node = torch.randint(high=self.n_idx, size=(self.n_atom,), dtype=torch.long)
        node[is_0] = data0.node[is_0]
        node[is_1] = data1.node[is_1]
        node_vec = F.one_hot(data1.node, self.n_idx).to(torch.float) * (2*t/(1-t)) \
            + torch.full((self.n_atom, self.n_idx), fill_value=1/self.n_idx) * 2 \
            + F.one_hot(node, self.n_idx).to(torch.float) * (-2/(1-t))

        # coord
        coord = data0.coord*(1-t)+data1.coord*t
        coord_vec = data1.coord - data0.coord

        # output
        return MolData(node, coord), MolVec(node_vec, coord_vec)

class MolFMDataset(Dataset):
    def __init__(self, mol_data: Dataset[Chem.Mol], dater: MolDater, t_sampler: TSampler):
        self.mol_data = mol_data
        self.dater = dater
        self.t_sampler = t_sampler
    def __len__(self):
        return len(self.mol_data)
    
    def __getitem__(self, idx):
        mol = self.mol_data[idx]

        data1 = self.dater.to_data(mol)
        data0 = self.dater.sample_init_data()
        t = self.t_sampler()
        data, vec = self.dater.interleave(data0, data1, t)
        return data, t, vec

class ExceptNoneDataset[T](Dataset[T]):
    logger = getLogger(f"{__module__}.{__qualname__}")

    def __init__(self, dataset: Dataset[T]):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx: int):
        if idx < 0 or len(self) <= idx:
            raise IndexError
        try:
            return self.dataset[idx]
        except Exception as e:
            

logger = get_logger(stream=True)
add_file_handler(logger, "train_mol/results/test/debug.log")
multiprocessing.set_start_method('fork')

dater = MolDater()
t_sampler = UniformTSampler(n_t_step=100)
data = UniMolLigandDataset('train', 'rdkit')
data = MolFMDataset(data, dater, t_sampler)
idx_sampler = InfiniteRandomSampler(data)
loader = DataLoader(data, batch_size=128, sampler=idx_sampler, num_workers=16, collate_fn=lambda x: x)
data_iter = iter(loader)

model = GraphFMModel(dater.n_idx)
optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4)
criterion = MolVecCriterion()
streamer = Streamers([
    LogStepStreamer(logger, AmpRange(10, 10000)), 
    SaveModelStreamer("train_mol/results/test/models/{step}", RepeatRange(10000))
])
stop_criterion = StepStopCriterion(10000)

train_fm(model, optimizer, data_iter, criterion, streamer, stop_criterion)



