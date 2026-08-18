import os
from argparse import Namespace, ArgumentParser
import yaml
import numpy as np
import torch
from src.fm.train import GStreamer, generate
from .data import MolEncoder, MolData
from .path import DiscDenoisePath, LinearDenoisePath, MolPath, Kappa
from .train.__main__ import MolFMModel, MaskedAtomDist, RandomCoordDist

class MolGStreamer[NBP, CBP](GStreamer[MolData, tuple[NBP, CBP]]):
    def __init__(self, node_path, coord_path):
        self.node_path = node_path
        self.coord_path = coord_path
        self.nodes = []
        self.coords = []

    def init(self, data, batch_idx):
        self.batch_idx = batch_idx
        self.nodes.append(data.node.cpu().numpy().copy())
        self.coords.append(data.coord.cpu().numpy().copy())
        
    def put(self, data, bpred, t, delta_t):
        self.nodes.append(data.node.cpu().numpy().copy())
        self.coords.append(data.coord.cpu().numpy().copy())
    
    def end(self):
        os.makedirs(os.path.dirname(self.node_path), exist_ok=True)
        np.save(self.node_path, np.stack(self.nodes))
        os.makedirs(os.path.dirname(self.coord_path), exist_ok=True)
        np.save(self.coord_path, np.stack(self.coords))


class GenScheduler:
    def __init__(self,
            kappa: Kappa,
            a: float,
            b: float,
            max_delta_t: float,
            eps_start: float, 
            eps_end: float):
        self.kappa = kappa
        self.a = a
        self.b = b
        self.max_delta_t = max_delta_t
        self.eps_start = eps_start
        self.eps_end = eps_end
    
    def iter_ts(self):
        t = self.eps_start
        while True:
            k, dk = self.kappa(t)
            alpha =1+t**self.a*(1-t)*self.b
            beta = alpha-1
            delta_t = min(self.max_delta_t, 1/(alpha*dk/(1-k)+beta*dk/k))
            yield t, delta_t, alpha
            t += delta_t
            if t >= 1-self.eps_end:
                break

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--studyname', required=True)
    parser.add_argument('--step', type=int, default=10000)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dir = f"mol/train/results/{args.studyname}"
    with open(f"{train_dir}/args.yaml") as f:
        targs = Namespace(**yaml.safe_load(f))
    gen_dir = f"mol/generate/train/{args.studyname}/{args.step}"
    

    mencoder = MolEncoder(120)
    init_node_dist = MaskedAtomDist(mencoder)
    init_coord_dist = RandomCoordDist(mencoder, targs.init_coord_std)
    path = MolPath(
        DiscDenoisePath(init_node_dist, 0.0, 0.0),
        LinearDenoisePath(init_coord_dist, 0.0, 0.0),
        targs.coord_weight,
    )
    fm_model = MolFMModel(mencoder)
    fm_model.load_state_dict(torch.load(f"{train_dir}/models/{args.step}.pth"))
    fm_model.to(device)
    gstreamers = [GStreamer() for _ in range(100)]
    for idx in np.random.choice(100, size=3, replace=False):
        gstreamers[idx] = MolGStreamer(f"{gen_dir}/sample/node/{idx}.npy", f"{gen_dir}/sample/coord/{idx}.npy", )
    ts = list(GenScheduler(path.node_path.kappa, 0.25, 0.25, 0.01, 1e-3, 1e-3).iter_ts())
    datas = generate(fm_model, path, gstreamers, ts, 100, 100)