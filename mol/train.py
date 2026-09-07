import os
import itertools as itr
from argparse import ArgumentParser
from functools import partial
from collections.abc import Callable, Container
from logging import getLogger
from pathlib import Path as _Path
import numpy as np, pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
from rdkit import Chem
from scipy.spatial.transform import Rotation as R
from scipy.optimize import linear_sum_assignment
from egnn_pytorch import EGNN
from src.utils.logger import get_logger, add_file_handler
from src.utils.random import set_random_seed
from src.utils.path import cleardir
from src.fm.train import train_fm, Path, FMModel, Loss, Streamer
from src.fm.utils import Optimizer, Streamers, LogStepStreamer, SaveModelStreamer, SaveLossStreamer, SaveGradStreamer, AmpContainer, RepeatContainer, CatContainer, StepStopCriterion
from src.data import ErrorNoneDataset
from src.data.coord import get_random_rotation_matrix
from src.data.datasets.unimol import UniMolLigandDataset

# data_iter
Mol = tuple[Tensor, Tensor, Tensor]

with open(_Path(__file__).parent / "atoms.txt") as f:
    ATOMS = f.read().splitlines()
MAX_ABS_CHARGE = 4

class MolDataset(Dataset[tuple[Mol, Mol]|None]):
    logger = getLogger(f"{__module__}.{__qualname__}")
    def __init__(self, mol_data: Dataset[Chem.Mol], n_atom: int, init_coord_std: float, mask_init: bool):
        """
        Parameters
        ----------
        init_coord_std: Used for both padded atoms of data1 and all atoms of data0

        mask_init: bool
            If False, all atoms are randomly initialized.
        
        """

        ## constructor parameters
        self.mol_data = mol_data
        self.n_atom = n_atom
        self.init_coord_std = init_coord_std
        self.mask_init = mask_init

        ## rng
        self.rng = np.random.default_rng(0)

        ## atom-idx map
        self.atoms = ['PAD']+ATOMS
        if mask_init:
            self.atoms.append('MASK')
        self.atom2idx = {atom: i for i, atom in enumerate(self.atoms)}
        self.n_atom_idx = len(self.atoms)

        self.n_charge_idx = MAX_ABS_CHARGE*2+1

    def __getitem__(self, idx: int):

        # get raw mol
        mol = self.mol_data[idx]

        # add hydrogen
        mol = Chem.AddHs(mol)

        # get n_mol_atom: truncate extra atoms
        n_mol_atom = mol.GetNumAtoms()
        if n_mol_atom > self.n_atom:
            raise ValueError(f"{n_mol_atom=} > {self.n_atom=}")
        n_pad_atom = self.n_atom - n_mol_atom

        # data1 node
        atom1 = torch.tensor(
            [self.atom2idx[mol.GetAtomWithIdx(i).GetSymbol()] for i in range(n_mol_atom)]+[self.atom2idx['PAD']]*n_pad_atom, 
            dtype=torch.long
        )

        # data1 coord
        ## centerize & random rotation
        mol_coord = mol.GetConformer().GetPositions()[:n_mol_atom]
        mol_coord = mol_coord - np.mean(mol_coord, axis=0)
        mol_coord = np.matmul(mol_coord, get_random_rotation_matrix(self.rng))
        ## add padding
        mol_pad_coord = self.rng.normal(size=(n_pad_atom,3))*self.init_coord_std
        mol_pad_coord -= np.mean(mol_pad_coord, axis=0)
        coord1 = np.concatenate([mol_coord, mol_pad_coord])

        # data1 charge
        charge1 = torch.tensor([mol.GetAtomWithIdx(i).GetFormalCharge() for i in range(n_mol_atom)]+[0]*n_pad_atom, dtype=torch.long)
        if (max_charge:=torch.max(charge1).item()) > MAX_ABS_CHARGE:
            self.logger.warning(f"[{idx}] {max_charge=} > {MAX_ABS_CHARGE=}")
        if (min_charge:=torch.min(charge1).item()) < -MAX_ABS_CHARGE:
            self.logger.warning(f"[{idx}] {min_charge=} < {-MAX_ABS_CHARGE=}")
        charge1.clamp_(-MAX_ABS_CHARGE, MAX_ABS_CHARGE).add_(MAX_ABS_CHARGE)

        # data0 node
        if self.mask_init:
            atom0 = torch.tensor([self.atom2idx['MASK']]*self.n_atom, dtype=torch.long)
        else:
            atom0 = torch.tensor(self.rng.integers(0, self.n_atom_idx, size=self.n_atom))

        # data0 coord, charge
        coord0 = self.rng.normal(size=(self.n_atom, 3))*self.init_coord_std
        coord0 -= np.mean(coord0, axis=0)
        charge0 = torch.full((self.n_atom, ), MAX_ABS_CHARGE, dtype=torch.long)

        # OT permutation
        dist = np.sqrt((coord1**2).sum(axis=1)[:, np.newaxis]+(coord0**2).sum(axis=1)[np.newaxis,:] - np.matmul(coord1, coord0.T)*2) # [N1, N0]
        _, col_ind = linear_sum_assignment(dist)
        atom0 = atom0[col_ind]
        coord0 = coord0[col_ind]
        charge0 = charge0[col_ind]

        # OT rotation
        rotation, _ = R.align_vectors(coord1, coord0)
        coord0 = rotation.apply(coord0)

        # coord to tensor
        coord0 = torch.tensor(coord0, dtype=torch.float)
        coord1 = torch.tensor(coord1, dtype=torch.float)


        return (atom0, coord0, charge0), (atom1, coord1, charge1)

    def __len__(self):
        return len(self.mol_data)

    def idx2charge(self, idx: int):
        return idx-MAX_ABS_CHARGE

class PathSampleDataset[D, Tgt, BPred](Dataset):
    def __init__(self, dataset: Dataset[tuple[D, D]], path: Path[D, Tgt, BPred]):
        self.dataset = dataset
        self.path = path
        self.N = 100


    def __getitem__(self, idx):
        data0, data1 = self.dataset[idx]
        t = np.random.randint(0, self.N)
        data, tgt = self.path.sample(data0, data1, t/self.N, (t+1)/self.N)
        return data, t/self.N, tgt

    def __len__(self):
        return len(self.dataset)


# Path
class DiscPath[Tgt, BPred](Path[Tensor, Tgt, BPred]):
    """
    Path for atoms & charges    
    """
    def build_head(self, node_dim: int) -> Callable[[Tensor], BPred]:
        raise NotImplementedError

class DenoiseDiscPath(DiscPath[Tensor, Tensor]):
    """
    D: [L]
    Tgt: [L]
    BPred: [B, L, V] logits
    V = n_atom_idx
    
    """
    def __init__(self, n_atom_idx: int, kappa: Callable[[float], float]):
        self.n_atom_idx = n_atom_idx
        self.kappa = kappa
    def sample(self, data0, data1, t0, t1):
        k = self.kappa(t0)
        # data
        data = data0.clone().detach()
        is_data1 = torch.rand_like(data, dtype=torch.float) < k
        data[is_data1] = data1[is_data1]
        # target
        return data, data1
    def update(self, datas, bpred, t0, t1):
        k = self.kappa(t0)
        dk = self.kappa(t1)-k
        for idx, data in enumerate(datas):
            data1_mask = torch.rand_like(data, dtype=torch.float) < dk / (1-k)
            data[data1_mask] = torch.multinomial(F.softmax(bpred[idx][data1_mask], dim=-1), num_samples=1).squeeze(-1)
        return datas
    def build_head(self, node_dim):
        return nn.Sequential(
            nn.Linear(node_dim, 512), 
            nn.GELU(), 
            nn.Linear(node_dim, self.n_atom_idx)
        )
    def build_criterion(self):
        return _DenoiseDiscCriterion()

class _DenoiseDiscCriterion(nn.Module):
    def forward(self, targets: list[Tensor], bpred: Tensor):
        B, N, V = bpred.shape
        btarget = torch.cat(targets).to(bpred.device) # [B*N, ]
        bpred = bpred.reshape(B*N, V)
        loss = F.cross_entropy(bpred, btarget)
        return Loss([loss], ['loss'], [1.0])

class DenoiseCoordPath(Path[Tensor, Tensor, Tensor]):
    def __init__(self, kappa: Callable[[float], float]):
        self.kappa = kappa

    def sample(self, data0, data1, t0, t1):
        k = self.kappa(t0)
        data = data0*(1-k)+data1*k
        return data, data1

    def build_criterion(self):
        return _DenoiseCoordCriterion()

class _DenoiseCoordCriterion(nn.Module):
    def forward(self, targets: list[Tensor], bpred: Tensor):
        btarget = torch.stack(targets).to(bpred.device)
        loss = F.mse_loss(bpred, btarget)
        return Loss([loss], ['loss'], [1.0])

class TuplePath(Path):
    def __init__(self, paths: list[Path], names: list[str]):
        self.paths = paths
        self.names = names
    def sample(self, data0, data1, t0, t1):
        outs = [path.sample(d0, d1, t0, t1) for path, d0, d1 in zip(self.paths, data0, data1)]
        return tuple(zip(*outs))
    def build_criterion(self):
        return _TupleCriterion([path.build_criterion() for path in self.paths], self.names)

class _TupleCriterion(nn.Module):
    def __init__(self, criteria: list[nn.Module], names: list[str]):
        super().__init__()
        self.criteria = criteria
        self.names = names
    def forward(self, targets, bpred):
        """
        targets: [n_data, n_path]
        bpred: [n_path]
        
        """
        targets = list(zip(*targets)) # [n_path, n_data]
        losses = []
        for i in range(len(self.criteria)):
            loss = self.criteria[i](targets[i], bpred[i])
            loss.names = [self.names[i]+'_'+name for name in loss.names]
            losses.append(loss)
        return Loss.cat(*losses)

class MolPath(TuplePath):
    def __init__(self, atom_path: DiscPath, coord_path: Path, charge_path: DiscPath):
        super().__init__([atom_path, coord_path, charge_path], ["atom", "coord", "charge"])

def cubic_kappa(t: float, a: float, b: float):
    # 常に k'(t) >= 0 となる条件: 
    # 概ね -1 <= a <= 2, -1 <= b <= 2 の領域 (より少し大きい)
    # ... Appendix D. で探索していた範囲
    return t-t**2*(1-t)*a+t*(1-t)**2*b

# Model
class MolFMModel[MolTgt](FMModel[Mol, MolTgt]):
    def __init__(self, n_atom_idx: int, n_charge_idx: int, atom_head: nn.Module, charge_head: nn.Module):
        super().__init__()
        d_model = 512
        n_layer = 6

        # backbone
        self.layers = nn.ModuleList([EGNN(dim=d_model) for _ in range(n_layer)])
        # embedding, head
        self.atom_emb = nn.Embedding(n_atom_idx, d_model)
        self.charge_emb = nn.Embedding(n_charge_idx, d_model)
        self.atom_head = atom_head
        self.charge_head = charge_head

    def forward(self, datas, ts):
        device = self.device()
        atoms, coords, charges = zip(*datas)
        atoms = torch.stack(atoms).to(device)
        coords = torch.stack(coords).to(device)
        charges = torch.stack(charges).to(device)
        coords0 = coords

        x_node = self.atom_emb(atoms)+self.charge_emb(charges)
        for layer in self.layers:
            x_node, coords = layer(x_node, coords)
        d_coords = coords - coords0 # [N, L] translation invariant
        d_coords = d_coords - torch.mean(d_coords, dim=0)

        return self.atom_head(x_node), d_coords, self.charge_head(x_node)

    def device(self) -> torch.device:
        return next(self.parameters()).device

class SaveBatchStreamer(Streamer[Mol, tuple[Tensor, Tensor, Tensor]]):
    def __init__(self, dir_format: str, steps: Container[int], n_sample_per_step: int, mdata: MolDataset):
        self.dir_format = dir_format
        self.steps = steps
        self.n_sample_per_step = n_sample_per_step
        self.mdata = mdata
        self.step = 0
    def put_data(self, batch):
        self.step += 1
        if self.step not in self.steps:
            return
        if len(batch) > self.n_sample_per_step:
            idxs = np.random.choice(len(batch), size=self.n_sample_per_step, replace=False)
        else:
            idxs = np.arange(len(batch))
        batch = [batch[idx] for idx in idxs]
        _, ts, _ = zip(*batch)
        dir = self.dir_format.format(step=self.step)
        os.makedirs(dir, exist_ok=True)
        for idx, data in zip(idxs, batch):
            (atom, coord, charge), t, (atom_tgt, coord_tgt, charge_tgt) = data
            pd.DataFrame({
                'atom': [self.mdata.atoms[a] for a in atom.tolist()], 
                'atom_tgt': [self.mdata.atoms[a] for a in atom_tgt.tolist()], # Discreteであることを前提としている。。
                'charge': [self.mdata.idx2charge(c) for c in charge.tolist()],
                'charge_tgt': [self.mdata.idx2charge(c) for c in charge_tgt.tolist()], 
                **{f'coord_{i}': coord[:,i] for i in range(3)}, 
                **{f'coord_tgt_{i}': coord_tgt[:,i] for i in range(3)}
            }).to_csv(f"{dir}/{idx}.tsv", sep='\t', index=False)
        pd.DataFrame({'idx': idxs, 't': ts}).to_csv(f"{dir}/t.tsv", sep='\t', index=False)

if __name__ == '__main__':

    # parameters
    parser = ArgumentParser()
    parser.add_argument("--studyname", required=True)
    args = parser.parse_args()
    batch_size = 64
    lr = 3e-4 * batch_size / 512 # original: batch_size=512, max_lr=3e-4
    n_atom = 80
    init_coord_std = 3.0

    # training
    result_dir = f"mol/results/{args.studyname}"
    cleardir(result_dir)
    logger = get_logger(stream=True)
    add_file_handler(logger, f"{result_dir}/debug.log")
    set_random_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    # data_iter
    dataset = UniMolLigandDataset('train', 'rdkit')
    dataset = mdata = MolDataset(dataset, n_atom, init_coord_std, mask_init=False)

    # path
    path = MolPath(
        atom_path:=DenoiseDiscPath(mdata.n_atom_idx, partial(cubic_kappa, a=1, b=-1)),
        coord_path:=DenoiseCoordPath(partial(cubic_kappa, a=0, b=0)),
        charge_path:=DenoiseDiscPath(mdata.n_charge_idx, partial(cubic_kappa, a=1, b=-1))
    )

    # data2: PathSample
    dataset = PathSampleDataset(dataset, path)
    dataset = ErrorNoneDataset(dataset)
    data_loader = DataLoader(dataset, batch_size=None, shuffle=True)
    data_iter = itr.chain.from_iterable(itr.repeat(data_loader))
    data_iter = itr.filterfalse(lambda x: x is None, data_iter)
    data_iter = itr.batched(data_iter, batch_size)

    # model
    model = MolFMModel(mdata.n_atom_idx, mdata.n_charge_idx, atom_path.build_head(512), charge_path.build_head(512)).to(device)
    criterion = path.build_criterion()

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = Optimizer(
        optimizer=optim, 
        scheduler=LambdaLR(optim, lambda step: step/5000 if step < 5000 else 1/(step/5000)**0.5), # original: warmup=2500
        clip_grad_norm=1.0
    )

    # other
    streamer = Streamers([
        LogStepStreamer(logger, AmpContainer(1, 10000)), 
        SaveModelStreamer(result_dir+"/models/{step}.pth", RepeatContainer(10000)),
        SaveLossStreamer(result_dir+"/loss.csv"),
        SaveGradStreamer(result_dir+"/grads/{step}/{k}.pth", CatContainer([1], AmpContainer(100, 10000))),
        SaveBatchStreamer(result_dir+"/sample_data/{step}",range(10), 3, mdata)
    ])
    stop_criterion = StepStopCriterion(10000)

    with torch.autocast('cuda', dtype=torch.bfloat16):
        train_fm(model, optimizer, data_iter, criterion, streamer, stop_criterion)

