import os, math, yaml
import itertools as itr
from argparse import ArgumentParser, Namespace
from copy import deepcopy
import numpy as np
import torch
from rdkit import Chem
from .train import MolDataset, get_mol_path, get_model, get_ts
from src.data.datasets.unimol import UniMolLigandDataset
from src.utils.logger import get_logger
from src.chem import atoms_coords_to_mol

if __name__ == '__main__':
    # arguments
    parser = ArgumentParser()
    parser.add_argument("--studyname", required=True)
    parser.add_argument("--step", type=int, default=10000)
    parser.add_argument("--gname", required=True)
    args = parser.parse_args()
    n_gen = 7
    batch_size = 128
    T = 100

    # environment
    gen_dir = f"mol/generates/{args.gname}/{args.studyname}/{args.step}"
    os.makedirs(f"{gen_dir}/mols", exist_ok=True)
    logger = get_logger(stream=True)
    train_dir = f"mol/trains/{args.studyname}"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(f"{train_dir}/args.yaml") as f:
        targs = Namespace(**yaml.safe_load(f))
    n_atom = targs.n_atom
    init_coord_std = targs.init_coord_std

    # mdata (encoder), path
    dataset = UniMolLigandDataset('train', 'rdkit')
    mdata = MolDataset(dataset, n_atom, init_coord_std, mask_init=args.mask_init)
    path = get_mol_path(mdata, targs)
    atom_path, coord_path, charge_path = path.paths
    ts = get_ts(targs)

    # model
    model = get_model(targs, mdata, path).to(device)
    model.load_state_dict(torch.load(f"{train_dir}/models/{args.step}.pth", map_location=device))
    model.eval()

    processes = [] # [B, T, P, D]
    for i_step in range(math.ceil(n_gen/batch_size)):
        B = min(batch_size, n_gen-i_step*batch_size)
        datas = [mdata.sample0() for b in range(B)]
        bprocess = [deepcopy(datas)] # [T, B, P, D]
        for s in range(len(ts)-1):
            
            t0 = ts[s]
            t1 = ts[s+1]
            with torch.inference_mode():
                bpred = model(datas, [t0]*B)
            datas = path.update(datas, bpred, t0, t1) # [B, P, D]
            bprocess.append(deepcopy(datas))

        processes += list(zip(*bprocess)) # [B, T, P, D]
    atom, coord, charge = zip(*itr.chain(*processes)) # [P, B*T, D]
    atoms = torch.stack(atom).reshape(B, T+1, n_atom).numpy()
    coords = torch.stack(coord).reshape(B, T+1, n_atom, 3).numpy()
    charges = torch.stack(charge).reshape(B, T+1, n_atom).numpy()
    np.save(f"{gen_dir}/atom.npy", atoms)
    np.save(f"{gen_dir}/coord.npy", coords)
    np.save(f"{gen_dir}/charge.npy", charges)

    logger.info("Parsing...")
    for b in range(B):
        atom = atoms[b,-1]
        coord = coords[b,-1]
        charge = charges[b,-1]
        mask = atom != mdata.atom2idx['PAD']
        atom = [mdata.atoms[a] for a in atom[mask]]
        coord = coord[mask]
        try:
            mol = atoms_coords_to_mol(atom, coord, 'rdkit')
        except Exception as e:
            continue
        with open(f"{gen_dir}/mols/{b}.sdf", 'w') as f:
            f.write(Chem.MolToMolBlock(mol))
