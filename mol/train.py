import os, yaml, re
import itertools as itr
import multiprocessing as mp
from argparse import ArgumentParser, Namespace
from functools import partial
from collections.abc import Callable, Container
from collections import defaultdict
from logging import getLogger
from pathlib import Path as _Path
import numpy as np, pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, get_worker_info
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
from src.data.coord import get_random_rotation_matrix
from src.data.datasets.unimol import UniMolLigandDataset
from mol.graph_attn import get_dist, AbsCoordEmbedding, GraphAttnLayer

# Data
Mol = tuple[Tensor, Tensor, Tensor] # atoms, coords, charges

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
        coord1 = mol.GetConformer().GetPositions()[:n_mol_atom]
        coord1 = coord1 - np.mean(coord1, axis=0)
        coord1 = np.matmul(coord1, get_random_rotation_matrix(self.rng))
        ## add padding
        if n_pad_atom > 0:
            mol_pad_coord = self.rng.normal(size=(n_pad_atom,3))*self.init_coord_std
            mol_pad_coord -= np.mean(mol_pad_coord, axis=0)
            coord1 = np.concatenate([coord1, mol_pad_coord])

        # data1 charge
        charge1 = torch.tensor([mol.GetAtomWithIdx(i).GetFormalCharge() for i in range(n_mol_atom)]+[0]*n_pad_atom, dtype=torch.long)
        if (max_charge:=torch.max(charge1).item()) > MAX_ABS_CHARGE:
            self.logger.warning(f"[{idx}] {max_charge=} > {MAX_ABS_CHARGE=}")
        if (min_charge:=torch.min(charge1).item()) < -MAX_ABS_CHARGE:
            self.logger.warning(f"[{idx}] {min_charge=} < {-MAX_ABS_CHARGE=}")
        charge1.clamp_(-MAX_ABS_CHARGE, MAX_ABS_CHARGE).add_(MAX_ABS_CHARGE)

        # data0
        atom0, coord0, charge0 = self.sample0()
        coord0 = coord0.numpy()

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

    def sample0(self) -> Mol:
        # data0 node
        if self.mask_init:
            atom0 = torch.tensor([self.atom2idx['MASK']]*self.n_atom, dtype=torch.long)
        else:
            atom0 = torch.tensor(self.rng.integers(0, self.n_atom_idx, size=self.n_atom))

        # data0 coord, charge
        coord0 = self.rng.normal(size=(self.n_atom, 3))*self.init_coord_std
        coord0 -= np.mean(coord0, axis=0)
        charge0 = torch.full((self.n_atom, ), MAX_ABS_CHARGE, dtype=torch.long)
        coord0 = torch.tensor(coord0, dtype=torch.float)
        return atom0, coord0, charge0

class PathSampleDataset[D, Tgt, BPred](Dataset):
    def __init__(self, dataset: Dataset[tuple[D, D]], path: Path[D, Tgt, BPred], ts: list[float]):
        self.dataset = dataset
        self.path = path
        self.ts = ts
        assert (self.ts[0], self.ts[-1]) == (0.0, 1.0)

    def __getitem__(self, idx):
        data0, data1 = self.dataset[idx]
        t = np.random.choice(self.ts[:-1])
        data, tgt = self.path.sample(data0, data1, t)
        return data, t, tgt

    def __len__(self):
        return len(self.dataset)

ERROR_FORMATS = {
    'large_mol': (ValueError, re.compile(r"n_mol_atom=(\d+) > self\.n_atom=(\d+)")), 
    'no_sn': (KeyError, re.compile(r"Sn")),
}
class ErrorNoneDataset(Dataset):
    logger = getLogger(f"{__module__}.{__qualname__}")

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.error_count = defaultdict(int)
        self.n = 0
        self.log_ns = AmpContainer(1000)
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx: int):
        if idx < 0 or len(self) <= idx:
            raise IndexError
        try:
            item = self.dataset[idx]
        except RuntimeWarning as e:
            raise e
        except Exception as e:
            for ename, (ecls, earg) in ERROR_FORMATS.items():
                if isinstance(e, ecls) and re.fullmatch(earg, e.args[0]):
                    self.error_count[ename] += 1
                    break
            else:
                self.logger.info(f"Unknown error at {type(self.dataset).__name__}[{idx}]: {type(e).__name__}{e.args}")
            item = None

        self.n += 1
        if self.n in self.log_ns:
            worker_info = get_worker_info()
            self.logger.debug(f"Error in {self.n} __getitem__:")
            for ename, ecount in sorted(self.error_count.items(), key=lambda x: -x[1]):
                self.logger.debug(f"    {ename}: {ecount}")

        return item

# Path
class DenoiseDiscPath(Path[Tensor, Tensor, Tensor]):
    """
    D: [L]
    Tgt: [L]
    BPred: [B, L, V] logits
    V = n_atom_idx
    
    """
    def __init__(self, n_atom_idx: int, kappa: Callable[[float], float]):
        self.n_atom_idx = n_atom_idx
        self.kappa = kappa
    def sample(self, data0, data1, t):
        k = self.kappa(t)
        # data
        data = data0.clone().detach()
        is_data1 = torch.rand_like(data, dtype=torch.float) < k
        data[is_data1] = data1[is_data1]
        # target
        return data, data1
    def update(self, datas, bpred, t0, t1):
        bpred = bpred.to(datas[0].device)
        k = self.kappa(t0)
        dk = self.kappa(t1)-k
        for idx, data in enumerate(datas):
            data1_mask = torch.rand_like(data, dtype=torch.float) < dk / (1-k)
            prob = F.softmax(bpred[idx][data1_mask], dim=-1)
            data[data1_mask] = torch.multinomial(prob, num_samples=1).squeeze(-1)
        return datas
    def criterion(self, targets: list[Tensor], bpred: Tensor):
        B, N, V = bpred.shape
        btarget = torch.cat(targets).to(bpred.device) # [B*N, ]
        bpred = bpred.reshape(B*N, V)
        loss = F.cross_entropy(bpred, btarget)
        return Loss([loss], ['loss'], [1.0])

class DenoiseCoordPath(Path[Tensor, Tensor, Tensor]):
    def __init__(self, kappa: Callable[[float], float]):
        self.kappa = kappa

    def sample(self, data0, data1, t):
        k = self.kappa(t)
        data = data0*(1-k)+data1*k
        return data, data1
    def update(self, datas, bpred, t0, t1):
        bpred = bpred.to(datas[0].device)
        k0 = self.kappa(t0)
        k1 = self.kappa(t1)
        datas = [data+(bpred[b]-data)*(k1-k0)/(1-k0) for b, data in enumerate(datas)]
        return datas
    def criterion(self, targets: list[Tensor], bpred: Tensor):
        btarget = torch.stack(targets).to(bpred.device)
        loss = F.mse_loss(bpred, btarget)
        return Loss([loss], ['loss'], [1.0])

class VectorCoordPath(Path[Tensor, Tensor, Tensor]):
    def __init__(self, kappa: Callable[[float], float], dt: float):
        self.kappa = kappa
        self.dt = dt
    def sample(self, data0, data1, t):
        k = self.kappa(t)
        k1 = self.kappa(t+self.dt)
        data = data0*(1-k)+data1*k
        vec = (data1-data0)*(k1-k)
        return data, vec
    def update(self, datas, bpred, t0, t1):
        bpred = bpred.to(datas[0].device)
        k0 = self.kappa(t0)
        k1 = self.kappa(t1)
        k0_dt = self.kappa(t0+self.dt)
        datas = [data+bpred[b]*(k1-k0)/(k0_dt-k0) for b, data in enumerate(datas)]
        return datas
    def criterion(self, targets, bpred):
        btarget = torch.stack(targets).to(bpred.device)
        loss = F.mse_loss(bpred, btarget)
        return Loss([loss], ['loss'], [1.0])

class TuplePath(Path):
    def __init__(self, paths: list[Path], names: list[str], weights: list[float]):
        self.paths = paths
        self.names = names
        self.weights = weights
    def sample(self, data0, data1, t):
        outs = [path.sample(d0, d1, t) for path, d0, d1 in zip(self.paths, data0, data1)]
        return tuple(zip(*outs))
    def update(self, datas, bpred, t0, t1):
        p2datas = list(zip(*datas))
        datas = [path.update(datas, bpred0, t0, t1) for path, datas, bpred0
                in zip(self.paths, p2datas, bpred)] # [P, B]
        return list(zip(*datas)) # [B, P]
    def criterion(self, targets, bpred):
        """
        targets: [n_data, n_path]
        bpred: [n_path]
        
        """
        targets = list(zip(*targets)) # [n_path, n_data]
        losses = [path.criterion(ts, bp) for path, ts, bp in zip(self.paths, targets, bpred)]
        loss = Loss.cat(losses, self.names, self.weights)
        return loss

class MolPath(TuplePath):
    def __init__(self, atom_path: Path, coord_path: Path, charge_path: Path, coord_weight: float):
        super().__init__([atom_path, coord_path, charge_path], ["atom", "coord", "charge"], [1, coord_weight, 1])

def cubic_kappa(t: float, a: float, b: float):
    """
    常に k'(t) >= 0 となる条件: 
        概ね -1 <= a <= 2, -1 <= b <= 2 の領域 (より少し大きい)
        ... Appendix D. で探索していた範囲
    """
    return t-t**2*(1-t)*a+t*(1-t)**2*b

# Model
## Head
class GraphHead(nn.Module):
    def __call__(self, x_node_inv: Tensor, x_node_equiv: Tensor|None, x_pair: Tensor|None, coord: Tensor):
        """
        Parameters
        ----------
        x_node_inv: [B, N, node_inv_size]
            rotation-invariant, translation-invariant
        x_node_equiv: [B, N, 3]
            rotation-equivariant, translation-equivariant
        x_pair: [B, N, N, pair_size]
            rotation, translation-invariant
        coord: [B, N, 3]
            input coord (rotation, translation-equivariant)

        Notes
        -----
        Some of the inputs can be None according to the backbone.
        Choose proper head dependent on the backbone.
        """
        return super().__call__(x_node_inv, x_node_equiv, x_pair, coord)

    def forward(self, x_node_inv: Tensor, x_node_equiv: Tensor|None, x_pair: Tensor|None, coord: Tensor):
        raise NotImplementedError
    
class NodeInvHead(nn.Sequential, GraphHead):
    def forward(self, x_node_inv, x_node_equiv, x_pair, coord):
        return super().forward(x_node_inv)

class RawCoordHead(GraphHead):
    def __init__(self, centerize: bool, t_inv: bool):
        super().__init__()
        self.centerize = centerize
    def forward(self, x_node_inv, x_node_equiv, x_pair, coord):
        if self.centerize:
            x_node_equiv = x_node_equiv-torch.mean(x_node_equiv, dim=1, keepdim=True)
        return x_node_equiv

class EquivCoordHead(nn.Sequential, GraphHead):
    def forward(self, x_node_inv, x_node_equiv, x_pair, coord):
        B, N, _ = coord.shape
        pair_coef = super().forward(x_pair) # [B, N, N, 1]
        coord_diff = coord.reshape(B, N, 1, 3) - coord.reshape(B, 1, N, 3) # [B, Na, Na, 3]
        coord_vecs = torch.sum(coord_diff * pair_coef, dim=2) / N # [B, Na, 3]
        return coord+coord_vecs

class TInvCoordHead(GraphHead):
    def __init__(self, equiv_coord_head: GraphHead):
        super().__init__()
        self.head = equiv_coord_head
    def forward(self, x_node_inv, x_node_equiv, x_pair, coord):
        equiv_coord = self.head(x_node_inv, x_node_equiv, x_pair, coord)
        return equiv_coord - coord

## Backbone
class MolBackbone(nn.Module):
    def __call__(self, atoms: Tensor, coords: Tensor, charges: Tensor, ts: Tensor) -> tuple[Tensor, Tensor|None, Tensor|None]:
        """
        Parameters
        ----------
        atoms: [B, N]
        coords: [B, N, 3]
        charges: [B, N]
        ts: [B,]

        Returns
        -------
        x_node_inv: [B, N, node_inv_size]
            rotation, translation-invariant feature
        x_node_equiv: optional, [B, N, 3]
            rotation, translation-equivariant feature
        x_pair: optional, [B, N, N, pair_size]
        """
        return super().__call__(atoms, coords, charges, ts)
    def forward(self, atoms: Tensor, coords: Tensor, charges: Tensor, ts: Tensor) -> tuple[Tensor, Tensor|None, Tensor|None]:
        raise NotImplementedError

### EGNN
class EGNNMolBackbone(MolBackbone):
    def __init__(self, n_atom_idx: int, n_charge_idx: int, d_model: int, n_layer: int, norm_coors: bool):
        super().__init__()

        # backbone
        self.layers = nn.ModuleList([EGNN(dim=d_model, norm_coors=norm_coors) for _ in range(n_layer)])
        # embedding, head
        self.atom_emb = nn.Embedding(n_atom_idx, d_model)
        self.charge_emb = nn.Embedding(n_charge_idx, d_model)
        self.t_emb = nn.Linear(1, d_model)

    def forward(self, atoms, coords, charges, ts):
        x_node = self.atom_emb(atoms)+self.charge_emb(charges)
        t_emb = self.t_emb(ts.unsqueeze(1)).unsqueeze(1) # [B,] -> [B,1] -> [B, 512] -> [B, 1(N), 512]
        x_node = x_node + t_emb
        for i, layer in enumerate(self.layers):
            x_node, coords = layer(x_node, coords)
        return x_node, coords, None

## Attention
class AtomPairEmbedding(nn.Module):
    def __init__(self, d_pair: int, n_atom_idx: int, n_charge_idx: int):
        super().__init__()
        emb_size = 128

        self.n_atom_idx = n_atom_idx
        self.atom_weight_emb = nn.Embedding(n_atom_idx**2, emb_size)
        self.atom_bias_emb = nn.Embedding(n_atom_idx**2, emb_size)
        self.n_charge_idx = n_charge_idx
        self.charge_weight_emb = nn.Embedding(n_charge_idx**2, emb_size)
        self.charge_bias_emb = nn.Embedding(n_charge_idx**2, emb_size)
        self.means = nn.Parameter(torch.zeros((emb_size,), dtype=torch.float))
        self.stds = nn.Parameter(torch.ones((emb_size,), dtype=torch.float))
        self.linear = nn.Linear(emb_size, d_pair)
        # Initialization from Uni-Mol
        # nn.init.uniform_(self.means, 0, 3)
        # nn.init.uniform_(self.stds, 0, 3)
        # nn.init.constant_(self.atom_weight_emb.weight, 1)
        # nn.init.constant_(self.atom_bias_emb.weight, 0)
        # nn.init.constant_(self.charge_weight_emb.weight, 1)
        # nn.init.constant_(self.charge_bias_emb.weight, 0)
                
        # Initialization in 3dVAE
        nn.init.normal_(self.atom_weight_emb.weight, 0.0, 1.0)
        nn.init.normal_(self.atom_bias_emb.weight, 0.0, 1.0)
        nn.init.normal_(self.charge_weight_emb.weight, 0.0, 1.0)
        nn.init.normal_(self.charge_bias_emb.weight, 0.0, 1.0)
        
    def forward(self, atoms: Tensor, charges: Tensor, coord: Tensor) -> Tensor:
        """
        Parameters
        ----------
        atoms: (long)[B, Na]
        coord: (float)[B, Na, 3]
        """

        B, Na = atoms.shape
        
        atom_pair_type = (atoms.reshape(B, Na, 1)*self.n_atom_idx+atoms.reshape(B, 1, Na))
        charge_pair_type = (charges.reshape(B, Na, 1)*self.n_charge_idx+charges.reshape(B, 1, Na))
        dist_weight = self.atom_weight_emb(atom_pair_type)+self.charge_weight_emb(charge_pair_type) # [B, Na, Na, Dpair]
        dist_bias = self.atom_bias_emb(atom_pair_type)+self.charge_bias_emb(charge_pair_type) # [B, Na, Na, Dpair]
        dist = get_dist(coord) # [B, Na, Na]
        pair_g = dist.unsqueeze(-1) * dist_weight + dist_bias
        stds = self.stds.abs() + 1e-5
        pair_dist_emb = torch.exp(-0.5*(((pair_g-self.means)/stds)**2)) / ((2*torch.pi)**0.5*stds)
        pair_emb = self.linear(pair_dist_emb)
        return pair_emb

class AttnMolBackbone(MolBackbone):
    def __init__(self, n_atom_idx: int, n_charge_idx: int, d_model: int, n_layer: int, n_head: int, abs_coord_emb: bool):
        super().__init__()
        self.n_head = n_head
        
        # embedding        
        self.atom_emb = nn.Embedding(n_atom_idx, d_model)
        self.pair_emb = AtomPairEmbedding(n_head, n_atom_idx, n_charge_idx)
        self.charge_emb = nn.Embedding(n_charge_idx, d_model)
        self.t_emb = nn.Linear(1, d_model)
        self.abs_coord_emb = AbsCoordEmbedding(d_model) if abs_coord_emb else lambda x: 0

        # layers
        self.layers = nn.ModuleList(GraphAttnLayer(d_model, n_head) 
                for _ in range(n_layer))


    def forward(self, atoms, coords, charges, ts):
        # Embedding
        x_node = self.atom_emb(atoms) \
                + self.charge_emb(charges) \
                + self.abs_coord_emb(coords) \
                + self.t_emb(ts.unsqueeze(-1)).unsqueeze(-2) # [B, Na, D]
        x_pair = x_pair_0 = self.pair_emb(atoms, charges, coords) # [B, Na(Q), Na(K), Dh]

        # Main
        B, N, _ = x_node.shape
        x_node_shaped = x_node.permute(1, 0, 2)
        x_pair_shaped = x_pair.permute(0, 3, 1, 2).reshape(B*self.n_head, N, N) # [B*Dh, Q, K]
        for i, layer in enumerate(self.layers):
            x_node_shaped, x_pair_shaped = layer(x_node_shaped, x_pair_shaped)
        x_pair_final = x_pair_shaped.reshape(B, self.n_head, N, N).permute(0, 2, 3, 1)
        x_node = x_node_shaped.permute(1, 0, 2)

        return x_node, None, x_pair_final-x_pair_0

## FMModel
class MolFMModel(FMModel[Mol, tuple[Tensor, Tensor, Tensor]]):
    def __init__(self, backbone: MolBackbone, atom_head: GraphHead, coord_head: GraphHead, charge_head: GraphHead):
        super().__init__()
        self.backbone = backbone
        self.heads = nn.ModuleList([atom_head, coord_head, charge_head])

    def forward(self, datas, ts):
        device = self.device()
        atoms, coords, charges = zip(*datas)
        atoms = torch.stack(atoms).to(device)
        coords = torch.stack(coords).to(device) # [B, N, 3]
        charges = torch.stack(charges).to(device)
        ts = torch.tensor(ts, dtype=torch.float).to(device) # [B,] dtype is necessary
        x_node_inv, x_node_equiv, x_pair = self.backbone(atoms, coords, charges, ts)
        return tuple(head(x_node_inv, x_node_equiv, x_pair, coords) for head in self.heads)
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

# Training
def get_ts(args: Namespace) -> list[float]:
    if args.t_scheduler == 'linear':
        return np.linspace(0, 1, args.t_n+1).tolist()
    elif args.t_scheduler == 'last_pow':
        """
        Parameters: a, b
        t(s) = 
            c*s when 0≦s≦b
            1-d*(1-s)^a when b≦s≦1
        c and d is determined so that t(s) is smooth.
        """
        a = args.t_a if args.t_a is not None else 2
        b = args.t_b if args.t_b is not None else 0.8
        c = a/(1-b+a*b)
        d = 1/(1-b)**(a-1)/(1-b+a*b)
        return [s*c if s <= b else 1-d*(1-s)**a 
                for s in np.linspace(0, 1, args.t_n+1).tolist()]
    else:
        raise ValueError(f"{args.t_scheduler=}")

def get_mol_path(mdata: MolDataset, args: Namespace):
    ## coord_path
    # if not hasattr(args, 'coord_path'): # temporary
    #     args.coord_path = 'denoise'
    coord_kappa = partial(cubic_kappa, a=0, b=0)
    if args.coord_path == 'denoise':
        coord_path = DenoiseCoordPath(coord_kappa)
    elif args.coord_path == 'vector':
        assert args.t_scheduler == 'linear'
        coord_path = VectorCoordPath(coord_kappa, 1/args.t_n)
    else:
        raise ValueError
    if getattr(args, 'disc_scheduler', 'sq') == 'sq':
        atom_path = DenoiseDiscPath(mdata.n_atom_idx, partial(cubic_kappa, a=1, b=-1))
        charge_path = DenoiseDiscPath(mdata.n_charge_idx, partial(cubic_kappa, a=1, b=-1))
    else:
        atom_path = DenoiseDiscPath(mdata.n_atom_idx, partial(cubic_kappa, a=0, b=0))
        charge_path = DenoiseDiscPath(mdata.n_charge_idx, partial(cubic_kappa, a=0, b=0))

    return MolPath(atom_path, coord_path, charge_path, 0.3)

def get_model(args: Namespace, mdata: MolDataset, path: MolPath) -> MolFMModel:
    if args.backbone == "egnn":
        backbone = EGNNMolBackbone(mdata.n_atom_idx, mdata.n_charge_idx, d_model=512, n_layer=6, norm_coors=args.norm_coors)
        coord_head = RawCoordHead(centerize=True)
    elif args.backbone == "attn":
        backbone = AttnMolBackbone(mdata.n_atom_idx, mdata.n_charge_idx, d_model=512, n_layer=8, n_head=64, abs_coord_emb=args.abs_coord_emb)
        if args.abs_coord_head:
            coord_head = NodeInvHead(nn.Linear(512, 512), nn.GELU(), nn.Linear(512, 3))
        else:
            coord_head = EquivCoordHead(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 3))
    else:
        raise ValueError(f"{args.backbone=}")
    if args.coord_path == 'vector' and not isinstance(coord_head, NodeInvHead):
        coord_head = TInvCoordHead(coord_head)
    atom_head = NodeInvHead(nn.Linear(512, 512), nn.GELU(), nn.Linear(512, mdata.n_atom_idx))
    charge_head = NodeInvHead(nn.Linear(512, 512), nn.GELU(), nn.Linear(512, mdata.n_charge_idx))
    return MolFMModel(backbone, atom_head, coord_head, charge_head)

def main():
    # parameters
    parser = ArgumentParser()
    parser.add_argument("--studyname", required=True)
    parser.add_argument("--num-workers", type=int, default=16)
    ## data
    parser.add_argument("--n-atom", type=int, default=80)
    parser.add_argument("--init-coord-std", type=float, default=3.0)
    parser.add_argument("--t-n", type=int, default=100)
    parser.add_argument("--t-scheduler", choices=['linear', 'last_pow'], default='linear')
    parser.add_argument("--t-a", type=float)
    parser.add_argument("--t-b", type=float)
    parser.add_argument("--disc-scheduler", choices=['linear', 'sq'], default='sq')
    parser.add_argument("--mask-init", action='store_true')
    parser.add_argument("--coord-path", choices=['denoise', 'vector'], default='denoise')
    ## model
    parser.add_argument("--backbone", choices=['egnn', 'attn'], default='attn')
    parser.add_argument("--norm-coors", action='store_true')
    ### attn
    parser.add_argument("--abs-coord-emb", action='store_true')
    parser.add_argument("--abs-coord-head", action='store_true')
    ## training
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-step", type=int, default=10000)
    parser.add_argument("--item-lr", type=float, default=3e-4/512) # original: batch_size=512, max_lr=3e-4
    args = parser.parse_args()

    # training
    result_dir = f"mol/trains/{args.studyname}"
    if os.path.exists(result_dir):
        raise ValueError(f"{result_dir=} already exists.")
    logger = get_logger(stream=True)
    add_file_handler(logger, f"{result_dir}/debug.log")
    set_random_seed(0)
    device = torch.device('cuda')
    with open(f"{result_dir}/args.yaml", 'w') as f:
        yaml.dump(vars(args), f, sort_keys=False)

    # data_iter
    dataset = UniMolLigandDataset('train', 'rdkit')
    dataset = mdata = MolDataset(dataset, args.n_atom, args.init_coord_std, mask_init=args.mask_init)

    # path
    path = get_mol_path(mdata, args)

    # data2: PathSample
    dataset = PathSampleDataset(dataset, path, get_ts(args))
    dataset = ErrorNoneDataset(dataset)
    data_loader = DataLoader(dataset, batch_size=None, shuffle=True, num_workers=args.num_workers)
    data_iter = itr.chain.from_iterable(itr.repeat(data_loader))
    data_iter = itr.filterfalse(lambda x: x is None, data_iter)
    data_iter = itr.batched(data_iter, args.batch_size)

    # model
    model = get_model(args, mdata, path).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=args.item_lr*args.batch_size)
    optimizer = Optimizer(
        optimizer=optim, 
        scheduler=LambdaLR(optim, lambda step: step/5000 if step < 5000 else 1/(step/5000)**0.5), # original: warmup=2500
        clip_grad_norm=1.0
    )

    # other
    streamer = Streamers([
        LogStepStreamer(logger, AmpContainer(1, 10000)), 
        SaveModelStreamer(result_dir+"/models/{step}.pth", RepeatContainer(0, 10000)),
        SaveLossStreamer(result_dir+"/loss.csv"),
        SaveGradStreamer(result_dir+"/grads/{step}/{k}.pth", CatContainer([1], AmpContainer(100, 10000))),
        SaveBatchStreamer(result_dir+"/sample_data/{step}",range(10), 3, mdata)
    ])
    stop_criterion = StepStopCriterion(args.max_step)

    train_fm(model, optimizer, data_iter, path, streamer, stop_criterion)

import warnings

if __name__ == '__main__':
    mp.set_start_method('fork')
    warnings.simplefilter('error')
    warnings.filterwarnings('ignore', 'numpy.core.numeric is deprecated and has been renamed to numpy._core.numeric. The numpy._core namespace contains private NumPy internals and its use is discouraged, as NumPy internals can change without warning in any release. In practice, most real-world usage of numpy.core is to access functionality in the public NumPy API. If that is the case, use the public NumPy API. If not, you are using NumPy internals. If you would still like to access an internal attribute, use numpy._core.numeric._frombuffer.', DeprecationWarning)
    main()