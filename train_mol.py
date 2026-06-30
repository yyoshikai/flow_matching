import itertools as itr
import multiprocessing
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from rdkit import Chem
from src.utils.logger import get_logger, add_file_handler
from src.data.data import ExceptNoneDataset
from src.data.datasets.unimol import UniMolLigandDataset
from src.data.sampler import InfiniteRandomSampler
from src.data.coord import get_random_rotation_matrix
from src.fm.train import Vec, TSampler, train_fm
from src.fm.utils import UniformTSampler, Streamers, SaveModelStreamer, LogStepStreamer, AmpRange, RepeatRange, StepStopCriterion, SaveDictLossStreamer
from src.mol.model import MolData, MolVec, GraphFMModel, MolVecCriterion
from src.mol.data import ATOMS
from src.utils.random import set_random_seed


class MolDater(Dataset[tuple[MolData, float, Vec[MolData]]]):
    def __init__(self):
        self.n_atom = 120
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

        coord = torch.tensor(np.concatenate([
            mol.GetConformer().GetPositions(),
            self.rng.normal(size=(n_no_atom, 3))*self.init_coord_std
        ]), dtype=torch.float32)
        return MolData(node, coord)

    def sample_init_data(self):
        node = torch.full((self.n_atom,), fill_value=self.mask_atom_idx, dtype=torch.long)
        coord = torch.tensor(self.rng.normal(size=(self.n_atom, 3)) * self.init_coord_std, dtype=torch.float32)
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

if __name__ == '__main__':

    logger = get_logger(stream=True)
    add_file_handler(logger, "train_mol/results/test/debug.log")
    multiprocessing.set_start_method('fork')
    set_random_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.debug(f"{device=}")

    dater = MolDater()
    t_sampler = UniformTSampler(n_t_step=100)
    data = UniMolLigandDataset('train', 'rdkit')
    data = MolFMDataset(data, dater, t_sampler)
    data = ExceptNoneDataset(data)
    idx_sampler = InfiniteRandomSampler(data)
    item_loader = DataLoader(data, batch_size=None, sampler=idx_sampler, num_workers=16)
    item_iter = itr.filterfalse(lambda x: x is None, iter(item_loader))
    batch_iter = itr.batched(item_iter, 128)

    model = GraphFMModel(dater.n_idx)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4)
    criterion = MolVecCriterion(coord_weight=3/dater.n_idx)
    streamer = Streamers([
        LogStepStreamer(logger, AmpRange(10, 10000)), 
        SaveModelStreamer("train_mol/results/test/models/{step}", RepeatRange(10000)), 
        SaveDictLossStreamer("train_mol/results/test/loss.csv")
    ])
    stop_criterion = StepStopCriterion(10000)

    train_fm(model, optimizer, batch_iter, criterion, streamer, stop_criterion)

