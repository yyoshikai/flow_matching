import itertools as itr
import multiprocessing
from argparse import Namespace, ArgumentParser
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LRScheduler

from rdkit import Chem
from src.utils.logger import get_logger, add_file_handler
from src.utils.path import cleardir
from src.utils.random import set_random_seed
from src.data.data import ExceptNoneDataset
from src.data.datasets.unimol import UniMolLigandDataset
from src.data.sampler import InfiniteRandomSampler
from src.data.coord import get_random_rotation_matrix
from src.fm.train import Vec, TSampler, train_fm
from src.fm.utils import *
from src.mol.model import MolData, MolVec, GraphFMModel, MolVecCriterion
from src.mol.data import ATOMS

class Kappas:
    def __init__(self, alpha: float):
        """
        alpha ノイズの最大割合
        """
        assert 0 <= alpha < 1
        self.alpha = alpha*4
    def __call__(self, t: float):
        k1 = self.alpha * t*(1-t)
        k0 = 1-t - k1*0.5
        k2 = t - k1*0.5
        dk1 = self.alpha*(1-2*t)
        dk0 = -1-dk1*0.5
        dk2 = 1-dk1*0.5
        return (k0, k1, k2), (dk0, dk1, dk2)


class MolDater(Dataset[tuple[MolData, float, Vec[MolData]]]):
    def __init__(self, n_atom: int, init_coord_std: float, end_coord_std: float, alpha: float):
        self.n_atom = n_atom
        self.init_coord_std = init_coord_std
        self.end_coord_std = end_coord_std

        self.no_atom_idx = 0
        self.atom2idx = {atom: i+1 for i, atom in enumerate(ATOMS)}
        self.mask_atom_idx = len(ATOMS)+1
        self.n_idx = len(ATOMS)+2
        self.rng = np.random.default_rng(0)
        self.kappas = Kappas(alpha)

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
        ks, dks = self.kappas(t)
        l = np.argmin([dk/k for k, dk in zip(ks, dks)])
        r = torch.rand((self.n_atom,))
        node = torch.randint(high=self.n_idx, size=(self.n_atom,), dtype=torch.long)
        node[r<=ks[0]] = data0.node[r<=ks[0]]
        node[r>ks[0]+ks[1]] = data1.node[r>ks[0]+ks[1]]
        node_vec = \
            F.one_hot(data0.node, self.n_idx).to(torch.float) * (dks[0]-ks[0]/ks[l]*dks[l]) \
            +F.one_hot(data1.node, self.n_idx).to(torch.float) * (dks[1]-ks[1]/ks[l]*dks[l]) \
            + torch.full((self.n_atom, self.n_idx), fill_value=1/self.n_idx) * (dks[2]-ks[2]/ks[l]*dks[l]) \
            + F.one_hot(node, self.n_idx).to(torch.float) * (dks[l]/ks[l])

        # coord
        coord = data0.coord*(1-t)+data1.coord*t
        coord_vec = data1.coord - data0.coord

        # output
        return MolData(node, coord), MolVec(node_vec, coord_vec)

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

    parser = ArgumentParser()
    parser.add_argument('--studyname', required=True)
    parser.add_argument('--init-coord-std', type=float, default=3.0)
    parser.add_argument('--end-coord-std', type=float, default=3.0)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1.0e-4)
    parser.add_argument('--max-step', type=int, default=10000)
    args = parser.parse_args()

    result_dir = f"train_mol/results/{args.studyname}"
    cleardir(result_dir)
    with open(f"{result_dir}/args.yaml", 'w') as f:
        yaml.dump(vars(args), f)
    logger = get_logger(stream=True)
    add_file_handler(logger, f"{result_dir}/debug.log")
    multiprocessing.set_start_method('fork')
    set_random_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.debug(f"{device=}")

    dater = MolDater(120, args.init_coord_std, args.end_coord_std, 0.1)
    t_sampler = UniformTSampler()
    data = UniMolLigandDataset('train', 'rdkit')
    data = MolFMDataset(data, dater, t_sampler)
    data = ExceptNoneDataset(data)
    idx_sampler = InfiniteRandomSampler(data)
    item_loader = DataLoader(data, batch_size=None, sampler=idx_sampler, num_workers=16)
    item_iter = itr.filterfalse(lambda x: x is None, iter(item_loader))
    batch_iter = itr.batched(item_iter, args.batch_size)

    model = GraphFMModel(dater.n_idx)
    model.to(device)
    optimizer = Optimizer(
        torch.optim.Adam(model.parameters(), lr=args.lr),
        scheduler=None, clip_grad_norm=1.0
    )
    criterion = MolVecCriterion(coord_weight=3/dater.n_idx/dater.init_coord_std)
    streamer = Streamers([
        LogStepStreamer(logger, AmpRange(10, 10000)), 
        SaveModelStreamer(result_dir+"/models/{step}.pth", RepeatRange(10000)), 
        SaveLossStreamer(result_dir+"/loss.csv"),
        SaveGradStreamer(result_dir+"/grads/{step}/{k}.pth", CatRange([1], AmpRange(100, 10000)))
    ])
    stop_criterion = StepStopCriterion(args.max_step)

    train_fm(model, optimizer, batch_iter, criterion, streamer, stop_criterion)

