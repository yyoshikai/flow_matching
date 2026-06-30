import os, math
from pathlib import Path
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..fm.train import Data, Vec, VecList, FMModel, Criterion
from ..fm.utils import DictLoss

@dataclass
class MolData(Data):
    node: Tensor # long[Na,]
    coord: Tensor # [Na, 3]
    def __post_init__(self):
        assert isinstance(self.node, Tensor)
        assert isinstance(self.coord, Tensor)
        Na, = self.node.shape
        assert self.coord.shape == (Na, 3)

@dataclass
class MolVec(Vec[MolData]):
    node: Tensor # [Na, T]
    coord: Tensor # [Na, 3]
    def __post_init__(self):
        assert isinstance(self.node, Tensor)
        assert isinstance(self.coord, Tensor)
        Na, T = self.node.shape
        assert self.coord.shape == (Na, 3)

@dataclass
class MolVecList(VecList[MolVec]):
    nodes: Tensor # [B, Na, T]
    coords: Tensor # [B, Na, T]

    def to_list(self):
        B, *_ = self.nodes.shape
        return [MolVec(self.nodes[i], self.coords[i]) for i in range(B)]

def get_dist(coord: Tensor):
    """
    Parameters
    ----------
    coord: (float)[B, Na, 3]

    Returns
    -------
    dist: (float)[B, Na, Na]
    """
    r2 = torch.sum(coord**2, dim=-1) # [*, Na]
    corr = torch.matmul(coord, coord.transpose(-1, -2)) # [*, Na, Na]
    dist2 = r2.unsqueeze(-1) + r2.unsqueeze(-2) - corr*2
    return torch.sqrt(dist2)

class GaussianPairEmbedding(nn.Module):
    def __init__(self, d_pair: int, n_node_type: int):
        super().__init__()
        self.n_node_type = n_node_type
        self.pair_gweight_emb = nn.Embedding(n_node_type**2, d_pair)
        self.pair_gbias_emb = nn.Embedding(n_node_type**2, d_pair)
        self.pair_gmean = nn.Parameter(torch.zeros((d_pair,), dtype=torch.float))
        self.pair_gstd = nn.Parameter(torch.ones((d_pair,), dtype=torch.float))
        # Initialization from Uni-Mol
        nn.init.uniform_(self.pair_gmean, 0, 3)
        nn.init.uniform_(self.pair_gstd, 0, 3)
        nn.init.constant_(self.pair_gweight_emb.weight, 1)
        nn.init.constant_(self.pair_gbias_emb.weight, 0)

    def forward(self, nodes: Tensor, coord: Tensor) -> Tensor:
        """
        Parameters
        ----------
        nodes: (long)[B, Na]
        coord: (float)[B, Na, 3]
        """

        B, Na = nodes.shape
        
        pair_type = (nodes.reshape(B, Na, 1)*self.n_node_type+nodes.reshape(B, 1, Na)).reshape(B, Na, Na)
        pair_gweight = self.pair_gweight_emb(pair_type) # [B, Na, Na, Dpair]
        pair_gbias = self.pair_gbias_emb(pair_type) # [B, Na, Na, Dpair]
        pair_dist = get_dist(coord) # [B, Na, Na]
        pair_g = pair_dist.unsqueeze(-1) * pair_gweight + pair_gbias
        pair_gstd = self.pair_gstd.abs() + 1e-5
        pair_dist_emb = torch.exp(-0.5*((pair_g-self.pair_gmean)/pair_gstd)**2) \
                / ((2*torch.pi)**0.5*pair_gstd)
        return pair_dist_emb

class GraphAttnLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff_factor=4, dropout=0.0):
        super().__init__()
        
        # self attention layer
        self.num_heads = num_heads
        Dh = d_model // num_heads
        layer_norm_eps = 1e-5
        assert Dh * num_heads == d_model, "d_model must be divisible by num_heads"
        self.in_proj = nn.Linear(d_model, 3*d_model)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.attn_dropout = nn.Dropout(dropout)
        
        # feed-forward layer
        d_ff = int(d_model*d_ff_factor)
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model, eps=layer_norm_eps),
            nn.Linear(d_model, d_ff), 
            nn.GELU(), 
            nn.Dropout(dropout), 
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout), 
        )
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x, edge):
        """
        Parameters
        ----------
        x(float)[L, B, D]
        edge(float)[B*Dh, L, L]

        Returns
        -------
        x: (float)[L, B, D]
        edge: (float)[B*Dh, L, L]

        B: batch_size
        L: L
        D: d_model
        Dh: head_dim
        H: num_heads

        """
        
        # set up shape vars
        num_heads = self.num_heads
        L, B, d_model = x.shape
        Dh = d_model // num_heads
        
        # residual connection
        x_res = x

        # pre layer_norm
        x = self.norm1(x)

        # attention
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        q = q.contiguous().view(L, B * num_heads, Dh).transpose(0, 1) # [B*H, L, Dh]
        k = k.contiguous().view(L, B * num_heads, Dh).transpose(0, 1) # [B*H, L, Dh]
        v = v.contiguous().view(L, B * num_heads, Dh).transpose(0, 1) # [B*H, L, Dh]
        t = torch.bmm(q, k.transpose(-2, -1)) / math.sqrt(Dh) # [B*H, L, L]

        attn_weights = F.softmax(t + edge, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.bmm(attn_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(L, B, d_model)
        attn_output = self.out_proj(attn_output)
        
        x = attn_output
        x = x_res+self.out_dropout(x)
        edge = t + edge

        # feed-forward
        x = x + self.ff(x)

        return x, edge

nn.TransformerEncoderLayer

class GraphFMModel(FMModel[MolData]):
    def __init__(self, n_node_type: int):
        super().__init__()
        d_model = 512
        num_layers = 8

        self.H = 64
        self.Dh = d_model // self.H
        self.node_emb = nn.Embedding(n_node_type, d_model)
        self.pair_emb = GaussianPairEmbedding(self.H, n_node_type)

        self.layers = nn.ModuleList(
            GraphAttnLayer(d_model, self.H) for _ in range(num_layers)
        )
        self.node_vec_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_node_type)
        )
        self.d_coord_proj = nn.Sequential(
            nn.Linear(self.H, self.H), 
            nn.GELU(), 
            nn.Linear(self.H, 1)
        )


    def forward(self, datas: list[MolData], ts: list[float]) -> list[MolVec]:
        device = self.device()
        
        nodes = torch.stack([data.node for data in datas]).to(device) # [B, Na]
        coord = torch.stack([data.coord for data in datas]).to(device) # [B, Na, 3]
        B, Na = nodes.shape

        # attention layers
        x_node = self.node_emb(nodes) # [B, Na, D]
        x_pair_0 = self.pair_emb(nodes, coord) # [B, Na(Q), Na(K), Dh]
        x_node_shaped = x_node.permute(1, 0, 2)
        x_pair_shaped = x_pair_0.permute(0, 3, 1, 2).reshape(B*self.H, Na, Na) # [B*Dh, Q, K]
        for i, layer in enumerate(self.layers):
            print(f"{i=} node={torch.sum(torch.isnan(x_node_shaped)).item()}, pair={torch.sum(torch.isnan(x_pair_shaped)).item()}")
            x_node_shaped, x_pair_shaped = layer(x_node_shaped, x_pair_shaped)
        x_pair_final = x_pair_shaped.reshape(B, self.H, Na, Na).permute(0, 2, 3, 1)
        x_node = x_node_shaped.permute(1, 0, 2)

        # node vector
        node_vecs = self.node_vec_proj(x_node) # [B, Na, Nt]

        # coord vector        
        d_x_pair = x_pair_final - x_pair_0 # [B, Na, Na, Dh]
        pair_coef = self.d_coord_proj(d_x_pair) # [B, Na, Na, 1]
        coord_diff = coord.reshape(B, Na, 1, 3) - coord.reshape(B, 1, Na, 3) # [B, Na, Na, 3]
        coord_vecs = torch.sum(coord_diff * pair_coef, dim=2) / Na # [B, Na, 3]

        return MolVecList(node_vecs, coord_vecs)

    def device(self) -> torch.device:
        return next(self.parameters()).device

class MolVecCriterion(Criterion[MolData, DictLoss]):
    def __init__(self, coord_weight: float):
        self.weights = {'node': 1, 'coord': coord_weight}

    def __call__(self, vecs_true: list[MolVec], vecs_pred: MolVecList):
        node_vecs_true = torch.stack([vec.node for vec in vecs_true]).to(vecs_pred.nodes.device)
        coord_vecs_true = torch.stack([vec.coord for vec in vecs_true]).to(vecs_pred.coords.device)

        return DictLoss({
            'node': torch.mean((node_vecs_true - vecs_pred.nodes)**2), 
            'coord': torch.mean((coord_vecs_true - vecs_pred.coords)**2)
        }, self.weights)
