import os, math
from collections.abc import Container
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Literal
import numpy as np, pandas as pd
import torch
from torch import Tensor
from rdkit import Chem
from src.data.coord import get_random_rotation_matrix
from src.fm.train import Streamer



with open(Path(__file__).parent / "atoms.txt") as f:
    ATOMS = f.read().splitlines()

# Mol
class Mol:
    pass
class ExpMol(Mol):
    def __init__(self, mol: Chem.Mol, rng: np.random.Generator, no_coord_std: float):
        self.mol = mol
        self.rng = rng
        self.no_coord_std = no_coord_std
class ImpMol(Mol):
    def __init__(self, atom_state: Literal['masked', 'random'], rng: np.random.Generator, coord_std: float):
        self.atom_state = atom_state
        self.rng = rng
        self.coord_std = coord_std

# MolData
@dataclass
class MolData:
    node: Tensor # long[Na,]
    coord: Tensor # [Na, 3]
    def __post_init__(self):
        assert isinstance(self.node, Tensor)
        assert isinstance(self.coord, Tensor)
        Na, = self.node.shape
        assert self.coord.shape == (Na, 3)

## Encode Mol -> MolData
class MolEncoder:
    logger = getLogger(f"{__module__}.{__qualname__}")
    def __init__(self, n_atom: int):
        self.n_atom = n_atom
        self.no_atom_idx = 0
        self.atoms = ['NO']+ATOMS+['MASK']
        self.atom2idx = {atom: i for i, atom in enumerate(self.atoms)}
        self.mask_atom_idx = self.atom2idx['MASK']
        self.n_idx = len(self.atoms)
    
    def encode(self, mol: Mol) -> MolData:
        if isinstance(mol, ExpMol):
            rdmol = mol.mol
            n_mol_atom = rdmol.GetNumAtoms()
            if n_mol_atom > self.n_atom:
                self.logger.warning(f"{n_mol_atom=} > {self.n_atom=}")
                n_mol_atom = self.n_atom
            n_no_atom = self.n_atom-n_mol_atom
            node = torch.tensor([self.atom2idx[rdmol.GetAtomWithIdx(i).GetSymbol()] for i in range(n_mol_atom)]+[self.no_atom_idx]*n_no_atom, dtype=torch.long)
            
            coord = rdmol.GetConformer().GetPositions() # [Na, 3]
            coord = coord - np.mean(coord, axis=0)
            coord = np.matmul(coord, get_random_rotation_matrix(mol.rng))

            coord = torch.tensor(np.concatenate([
                rdmol.GetConformer().GetPositions(),
                mol.rng.normal(size=(n_no_atom, 3))*mol.no_coord_std
            ]), dtype=torch.float32)
            return MolData(node, coord)
        elif isinstance(mol, ImpMol):
            node = torch.full((self.n_atom,), fill_value=self.mask_atom_idx, dtype=torch.long)
            coord = torch.tensor(mol.rng.normal(size=(self.n_atom, 3)) * mol.coord_std, dtype=torch.float32)
            return MolData(node, coord)
        else:
            raise ValueError

class SaveDataSampleStreamer(Streamer[MolData, tuple[Tensor, Tensor]]):
    def __init__(self, dir_format: str, mencoder: MolEncoder, n_sample: int, range: Container):
        self.dir_format = dir_format
        self.mencoder = mencoder
        self.range = range
        self.n_sample = n_sample
        self.step = 0
    def put_data(self, batch):
        self.step += 1
        if self.step not in self.range: 
            return
        dir = self.dir_format.format(step=self.step)
        os.makedirs(f"{dir}/batch", exist_ok=True)
        if len(batch) > self.n_sample:
            idxs = np.random.choice(len(batch), size=self.n_sample, replace=False)
        else:
            idxs = np.arange(len(batch))
        batch = [batch[idx] for idx in idxs]
        _, ts, _ = zip(*batch)
        for idx, data in zip(idxs, batch):
            mol, t, target = data
            node_tgt, coord_tgt = target
            data = {
                'atom': [self.mencoder.atoms[i] for i in mol.node.tolist()],
                'tgt_atom': [self.mencoder.atoms[i] for i in node_tgt.tolist()]
            }
            for l in range(3):
                data[f'coord_{l}'] = mol.coord[:,l]
            for l in range(3):
                data[f'tgt_coord_{l}'] = coord_tgt[:,l]
            df = pd.DataFrame(data)
            df.to_csv(f"{dir}/batch/{idx}.csv")
            df.to_string(f"{dir}/batch/{idx}.txt")
        pd.DataFrame({'idx': idxs, 't': ts}).to_csv(f"{dir}/t.csv", index=False)

