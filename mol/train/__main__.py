import itertools as itr
import multiprocessing
from argparse import Namespace, ArgumentParser
from copy import deepcopy
from typing import Literal
import yaml
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR

from rdkit import Chem
from src.utils.logger import get_logger, add_file_handler
from src.utils.path import cleardir
from src.utils.random import set_random_seed
from src.data import ExceptNoneDataset
from src.data.datasets.unimol import UniMolLigandDataset
from src.data.sampler import InfiniteRandomSampler
from src.fm.train import train_fm, Distribution, FMModel
from src.fm.utils import *
from ..data import ImpMol, ExpMol, MolEncoder, MolData, SaveDataSampleStreamer
from ..path import DiscDenoisePath, LinearDenoisePath, MolPath
from ..model import GraphAttnModel, GaussianPairEmbedding, TrigCoordEmbedding

# distribution
class InitMolDist(Distribution[MolData]):
    def __init__(self, 
            mol_encoder: MolEncoder, 
            atom_state: Literal['masked', 'random'], 
            seed: int, 
            coord_std: float
    ):
        self.mol_encoder = mol_encoder
        self.atom_state = atom_state
        self.rng = np.random.default_rng(seed)
        self.coord_std = coord_std
    def sample(self):
        self.rng.random()
        mol = ImpMol(self.atom_state, deepcopy(self.rng), self.coord_std)
        return self.mol_encoder.encode(mol)

class UniformTDist(Distribution[float]):
    def __init__(self, eps: float=1e-3):
        self.eps = eps
    def sample(self):
        return np.random.rand() * (1-self.eps)

# Dataset
class MolFMDataset[NT, CT, NBP, CBP](Dataset[tuple[MolData, float, tuple[NT, CT]]]):
    def __init__(self, 
            mol_data: Dataset[Chem.Mol],
            no_coord_std: float, 
            init_dist: Distribution[MolData],
            t_dist: Distribution[float],
            path: MolPath[NT, CT, NBP, CBP],
            mencoder: MolEncoder,
    ):
        self.mol_data = mol_data
        self.no_coord_std = no_coord_std
        self.t_dist = t_dist
        self.init_dist = init_dist
        self.path = path
        self.mencoder = mencoder
        self.rng = np.random.default_rng(0)

    def __getitem__(self, idx):
        self.rng.random()
        mol = ExpMol(self.mol_data[idx], self.rng, self.no_coord_std)
        data1 = self.mencoder.encode(mol)
        data0 = self.init_dist.sample()
        t = self.t_dist.sample()
        data, target = self.path.sample(data0, data1, t)
        return data, t, target
    
    def __len__(self):
        return len(self.mol_data)
    
class MolFMModel(FMModel[MolData, tuple[Tensor, Tensor]]):
    def __init__(self, mencoder: MolEncoder):
        super().__init__()
        n_node_type = mencoder.n_idx
        
        # graph model
        self.graph_model = GraphAttnModel()
        d_model = self.graph_model.d_model
        H = self.graph_model.H

        # embedding        
        self.node_emb = nn.Embedding(n_node_type, d_model)
        self.pair_emb = GaussianPairEmbedding(H, n_node_type)
        self.t_emb = nn.Linear(1, d_model)
        self.trig_coord_emb = TrigCoordEmbedding(d_model)
        
        # projection
        self.node_logit_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_node_type)
        )
        self.coord_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3)
        )

    def forward(self, datas: list[MolData], ts: list[float]):
        device = self.device()
        
        nodes = torch.stack([data.node for data in datas]).to(device) # [B, Na]
        coord = torch.stack([data.coord for data in datas]).to(device) # [B, Na, 3]
        ts = torch.tensor(ts, dtype=torch.float32).to(device)
        B, Na = nodes.shape

        # Embedding
        x_node = self.node_emb(nodes) \
                + self.trig_coord_emb(coord) \
                + self.t_emb(ts.unsqueeze(-1)).unsqueeze(-2) # [B, Na, D]
        x_pair_0 = self.pair_emb(nodes, coord) # [B, Na(Q), Na(K), Dh]
        
        # Main
        x_node, x_pair_final = self.graph_model(x_node, x_pair_0)
        
        # Projection
        node_logit = self.node_logit_proj(x_node) # [B, Na, Nt]
        coord_out = self.coord_proj(x_node)

        return node_logit, coord_out

    def device(self) -> torch.device:
        return next(self.parameters()).device

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--studyname', required=True)
    parser.add_argument('--init-coord-std', type=float, default=3.0)
    parser.add_argument('--no-coord-std', type=float, default=3.0)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1.5e-4) # 原論文: batch_size=512, max_lr=3e-4
    parser.add_argument('--max-step', type=int, default=10000)
    parser.add_argument('--coord-weight', type=float, default=0.001)
    args = parser.parse_args()

    result_dir = f"mol/train/results/{args.studyname}"
    cleardir(result_dir)
    with open(f"{result_dir}/args.yaml", 'w') as f:
        yaml.dump(vars(args), f)
    logger = get_logger(stream=True)
    add_file_handler(logger, f"{result_dir}/debug.log")
    multiprocessing.set_start_method('fork')
    set_random_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.debug(f"{device=}")

    # Dataset
    mencoder = MolEncoder(120)
    mol_data = UniMolLigandDataset('train', 'rdkit')
    init_dist = InitMolDist(mencoder, 'masked', 0, args.init_coord_std)
    t_dist = UniformTDist()
    path = MolPath(
        DiscDenoisePath(0.0, 0.0), 
        LinearDenoisePath(0.0, 0.0),
        args.coord_weight,
    )
    data = MolFMDataset(mol_data, args.no_coord_std, init_dist, t_dist, path, mencoder)
    data = ExceptNoneDataset(data)
    item_loader = DataLoader(data, batch_size=None, sampler=InfiniteRandomSampler(data), num_workers=16)
    item_iter = iter(item_loader)
    item_iter = itr.filterfalse(lambda x: x is None, item_iter)
    batch_iter = itr.batched(item_iter, args.batch_size)

    # Model    
    model = MolFMModel(mencoder)
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = Optimizer(
        optimizer=optim, 
        scheduler=LambdaLR(optim, lambda step: step/5000 if step < 5000 else 1/(step/5000)**0.5), # original: warmup=2500
        clip_grad_norm=1.0
    )

    # Training
    streamer = Streamers([
        LogStepStreamer(logger, AmpRange(10, 10000)), 
        SaveModelStreamer(result_dir+"/models/{step}.pth", RepeatRange(10000)),
        SaveLossStreamer(result_dir+"/loss.csv"),
        SaveGradStreamer(result_dir+"/grads/{step}/{k}.pth", CatRange([1], AmpRange(100, 10000))),
        SaveDataSampleStreamer(result_dir+"/sample_data/{step}", mencoder, 3, range(10))
    ])
    stop_criterion = StepStopCriterion(args.max_step)

    train_fm(model, optimizer, batch_iter, path, streamer, stop_criterion)
