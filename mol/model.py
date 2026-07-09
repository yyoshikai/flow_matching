import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

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
    return torch.sqrt(torch.abs(dist2))

class GaussianPairEmbedding(nn.Module):
    def __init__(self, d_pair: int, n_node_type: int):
        super().__init__()
        emb_size = 128

        self.n_node_type = n_node_type
        self.weight_emb = nn.Embedding(n_node_type**2, emb_size)
        self.bias_emb = nn.Embedding(n_node_type**2, emb_size)
        self.means = nn.Parameter(torch.zeros((emb_size,), dtype=torch.float))
        self.stds = nn.Parameter(torch.ones((emb_size,), dtype=torch.float))
        self.linear = nn.Linear(emb_size, d_pair)
        # Initialization from Uni-Mol
        # nn.init.uniform_(self.means, 0, 3)
        # nn.init.uniform_(self.stds, 0, 3)
        # nn.init.constant_(self.weight_emb.weight, 1)
        # nn.init.constant_(self.bias_emb.weight, 0)

        # Initialization in 3dVAE
        nn.init.normal_(self.weight_emb.weight, 0.0, 1.0)
        nn.init.normal_(self.bias_emb.weight, 0.0, 1.0)


    def forward(self, nodes: Tensor, coord: Tensor) -> Tensor:
        """
        Parameters
        ----------
        nodes: (long)[B, Na]
        coord: (float)[B, Na, 3]
        """

        B, Na = nodes.shape
        
        pair_type = (nodes.reshape(B, Na, 1)*self.n_node_type+nodes.reshape(B, 1, Na)).reshape(B, Na, Na)
        dist_weight = self.weight_emb(pair_type) # [B, Na, Na, Dpair]
        dist_bias = self.bias_emb(pair_type) # [B, Na, Na, Dpair]
        dist = get_dist(coord) # [B, Na, Na]
        pair_g = dist.unsqueeze(-1) * dist_weight + dist_bias
        stds = self.stds.abs() + 1e-5
        pair_dist_emb = torch.exp(-0.5*(((pair_g-self.means)/stds)**2)) \
                / ((2*torch.pi)**0.5*stds)
        pair_emb = self.linear(pair_dist_emb)
        return pair_emb


class TrigCoordEmbedding(nn.Module):
    """
    Embed coord directly with sinusoidal embedding
    ( = remove 3d-equivariance)
    
    """
    def __init__(self, D: int):
        super().__init__()
        assert D % 2 == 0
        self.D = D
        coef = torch.tensor([10/10000**(d*2/D) for d in range(D//2)])
        self.register_buffer('coef', coef)
        self.proj = nn.Linear(D*3, D)

    def forward(self, coord: Tensor):
        B, Na, _ = coord.shape
        x_sin = torch.sin(coord.unsqueeze(-1)*self.coef) # [B, Na, 3, D/2]
        x_cos = torch.sin(coord.unsqueeze(-1)*self.coef) # [B, Na, 3, D/2]
        x = torch.cat([x_sin, x_cos], dim=-1).reshape(B, Na, 3*self.D)
        x = self.proj(x)
        return x



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

class GraphAttnModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = 512
        self.num_layers = 8
        self.H = 64

        self.layers = nn.ModuleList(
            GraphAttnLayer(self.d_model, self.H) for _ in range(self.num_layers)
        )

    def forward(self, x_node: Tensor, x_pair: Tensor) -> tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        x_node: [B, Na, D]
        x_pair: [B, Na, Na, Dh]

        Returns
        -------
        x_node: [B, Na, D]
        x_pair: [B, Na, Na, Dh]
        
        """
        B, Na, _ = x_node.shape

        x_node_shaped = x_node.permute(1, 0, 2)
        x_pair_shaped = x_pair.permute(0, 3, 1, 2).reshape(B*self.H, Na, Na) # [B*Dh, Q, K]
        for i, layer in enumerate(self.layers):
            x_node_shaped, x_pair_shaped = layer(x_node_shaped, x_pair_shaped)
        x_pair = x_pair_shaped.reshape(B, self.H, Na, Na).permute(0, 2, 3, 1)
        x_node = x_node_shaped.permute(1, 0, 2)
        return x_node, x_pair


class DiffCoordHead(nn.Module):
    def __init__(self, D):
        self.proj = nn.Sequential(
            nn.Linear(D, D), 
            nn.GELU(), 
            nn.Linear(D, 1)
        )
    def forward(self, x: Tensor, coord: Tensor):
        """
        Parameters
        ----------
        x: Tensor(float) [B, Na, Na, D]
        coord: Tensor(float) [B, Na, 3]
        """
        B, Na, _ = coord.shape
        
        pair_coef = self.proj(x) # [B, Na, Na, 1]
        coord_diff = coord.reshape(B, Na, 1, 3) - coord.reshape(B, 1, Na, 3) # [B, Na, Na, 3]
        coord_vecs = torch.sum(coord_diff * pair_coef, dim=2) / Na # [B, Na, 3]
        return coord_vecs
