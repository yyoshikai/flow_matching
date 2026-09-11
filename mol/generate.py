import os, math, yaml
import itertools as itr
from argparse import ArgumentParser, Namespace
from copy import deepcopy
import numpy as np
import torch
from .train import get_mol_path, MolDataset, MolFMModel
from src.data.datasets.unimol import UniMolLigandDataset

if __name__ == '__main__':
    # arguments
    parser = ArgumentParser()
    parser.add_argument("--studyname", required=True)
    parser.add_argument("--step", type=int, default=10000)
    parser.add_argument("--gname", required=True)
    args = parser.parse_args()
    n_gen = 1
    batch_size = 128
    T = 100

    # environment
    gen_dir = f"mol/generates/{args.gname}/{args.studyname}/{args.step}"
    os.makedirs(gen_dir, exist_ok=True)
    train_dir = f"mol/trains/{args.studyname}"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(f"{train_dir}/args.yaml") as f:
        targs = Namespace(**yaml.safe_load(f))
    n_atom = targs.n_atom
    init_coord_std = targs.init_coord_std

    # mdata (encoder), path
    dataset = UniMolLigandDataset('train', 'rdkit')
    mdata = MolDataset(dataset, n_atom, init_coord_std, mask_init=False)
    path = get_mol_path(mdata)
    atom_path, coord_path, charge_path = path.paths

    # model
    model = MolFMModel(mdata.n_atom_idx, mdata.n_charge_idx, atom_path.build_head(512), charge_path.build_head(512)).to(device)
    model.load_state_dict(torch.load(f"{train_dir}/models/{args.step}.pth", map_location=device))
    model.eval()

    os.makedirs(f"{gen_dir}/ts", exist_ok=True)

    processes = [] # [B, T, P, D]
    for i_step in range(math.ceil(n_gen/batch_size)):
        B = min(batch_size, n_gen-i_step*batch_size)
        datas = [mdata.sample0() for b in range(B)]
        bprocess = [] # [T, B, P, D]
        for t in range(T):
            atom, coord, charge = zip(*datas)
            np.save(f"{gen_dir}/ts/{t}_atom.npy", torch.stack(atom).numpy())
            np.save(f"{gen_dir}/ts/{t}_coord.npy", torch.stack(coord).numpy())
            np.save(f"{gen_dir}/ts/{t}_charge.npy", torch.stack(charge).numpy())
            
            t0 = t/T
            t1 = (t+1)/T
            with torch.inference_mode():
                bpred = model(datas, [t0]*B)
            datas = path.update(datas, bpred, t0, t1) # [B, P, D]
            bprocess.append(deepcopy(datas))

        processes += list(zip(*bprocess)) # [B, T, P, D]
    atom, coord, charge = zip(*itr.chain(*processes)) # [P, B*T, D]
    np.save(f"{gen_dir}/atom.npy", torch.stack(atom).reshape(B, T, n_atom).numpy())
    np.save(f"{gen_dir}/coord.npy", torch.stack(coord).reshape(B, T, n_atom, 3).numpy())
    np.save(f"{gen_dir}/charge.npy", torch.stack(charge).reshape(B, T, n_atom).numpy())
