import os, math
from collections.abc import Container
from collections import namedtuple
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
MAX_ABS_CHARGE = 4

# Mol
class Mol:
    pass
MolData = tuple[Tensor, Tensor, Tensor] # [node, coord, charge]

class ExpMol(Mol):
    logger = getLogger(f"{__module__}.{__qualname__}")

    def __init__(self, mol: Chem.Mol, rng: np.random.Generator, no_coord_std: float):
        self.mol = mol
        self.rng = rng
        self.no_coord_std = no_coord_std
    def encode(self, n_atom, atom2idx: dict[str, int], no_atom_idx: int,)
        rdmol = self.mol
        n_mol_atom = rdmol.GetNumAtoms()
        if n_mol_atom > n_atom:
            self.logger.warning(f"{n_mol_atom=} > {n_atom=}")
            n_mol_atom = n_atom
        n_no_atom = n_atom-n_mol_atom
        node = torch.tensor([atom2idx[rdmol.GetAtomWithIdx(i).GetSymbol()] for i in range(n_mol_atom)]+[no_atom_idx]*n_no_atom, dtype=torch.long)
        
        coord = rdmol.GetConformer().GetPositions() # [Na, 3]
        coord = coord - np.mean(coord, axis=0)
        coord = np.matmul(coord, get_random_rotation_matrix(self.rng))

        coord = torch.tensor(np.concatenate([
            coord,
            self.rng.normal(size=(n_no_atom, 3))*self.no_coord_std
        ]), dtype=torch.float32)

        return (node, coord, charge)
class ImpMol(Mol):
    def __init__(self, atom_state: Literal['masked', 'random'], rng: np.random.Generator, coord_std: float):
        self.atom_state = atom_state
        self.rng = rng
        self.coord_std = coord_std

    def encode(self, n_atom, mask_atom_idx, ):
        node = torch.full((n_atom,), fill_value=mask_atom_idx, dtype=torch.long)
        coord = torch.tensor(self.rng.normal(size=(n_atom, 3)) * self.coord_std, dtype=torch.float32)
        return(node, coord, charge)
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
                coord,
                mol.rng.normal(size=(n_no_atom, 3))*mol.no_coord_std
            ]), dtype=torch.float32)
            return (node, coord, charge)
        elif isinstance(mol, ImpMol):
            node = torch.full((self.n_atom,), fill_value=self.mask_atom_idx, dtype=torch.long)
            coord = torch.tensor(mol.rng.normal(size=(self.n_atom, 3)) * mol.coord_std, dtype=torch.float32)
            return(node, coord, charge)
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

