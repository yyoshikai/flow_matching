import torch
import torch.nn.functional as F
from torch import Tensor
from src.fm.train import Path, Loss, Distribution
from .data import MolData


class Kappa:
    def __call__(self, t: float) -> tuple[float, float]:
        """
        Returns
        -------
        k(float): kappa
        d_k(float): dkappa/dt
        """
        raise NotImplementedError

class CubicKappa:
    def __init__(self, a: float, b: float):
        self.a = a
        self.b = b
        # 常に k'(t) >= 0 となる条件
        # 概ね -1 <= a <= 2, -1 <= b <= 2 の領域 (より少し大きい)
        # ... Appendix D. で探索していた範囲
        if a < -1 or b < -1:
            raise ValueError
        if a+2*b >= 0 and 2*a+b >= 0:
            if (a**2+a*b+b**2-3*a-3*b) > 0:
                raise ValueError
    def __call__(self, t):
        k = t - t**2*(1-t)*self.a + t*(1-t)**2*self.b
        dk = 1 + (3*t**2-2*t)*self.a + (3*t**2-4*t+1)*self.b
        assert 0 <= k <= 1 and 0 <= dk, f"{t=}, {k=}, {dk=}"
        return k, dk

class DiscDenoisePath(Path[Tensor, Tensor, Tensor]):
    def __init__(self, init_dist: Distribution[Tensor], a: float, b: float):
        self.init_dist = init_dist
        self.kappa = CubicKappa(a, b)
        self.rng = torch.Generator()

    def sample(self, data1, t):
        assert data1.dtype == torch.long
        k, dk = self.kappa(t)
        data = self.init_dist.sample().to(data1)
        is_data1 = torch.rand_like(data1, dtype=torch.float) < k
        data[is_data1] = data1[is_data1]
        return data, data1
    def sample_init(self):
        return self.init_dist.sample()
    def update(self, datas, bpred, t, delta_t, alpha):
        n = len(datas)
        k, dk = self.kappa(t)
        p_1 = delta_t * alpha * dk / (1-k)
        p_0 = delta_t * (alpha-1) * dk / k
        assert 0 <= p_1+p_0 <= 1, f"{p_0=}, {p_1=}"
        for idx, data in enumerate(datas):
            data0 = self.init_dist.sample()
            r = torch.rand_like(data, dtype=torch.float)
            data[r < p_1] = torch.multinomial(F.softmax(bpred[idx][r < p_1].to(data.device), dim=-1), num_samples=1).squeeze(-1)
            data[(p_1 <= r)& (r < p_1+p_0)] = data0[(p_1 <= r)& (r < p_1+p_0)]
        return datas
    def criterion(self, targets, bpred):
        """
        targets: list[Tensor(long)[N]]
        bpred: Tensor(float)[B, N, V]
            logits
        """        
        B, N, V = bpred.shape
        btarget = torch.cat(targets).to(bpred.device) # [B*N, ]
        bpred = bpred.reshape(B*N, V)
        loss = F.cross_entropy(bpred, btarget)
        return Loss([loss], ['loss'], [1.0])

class LinearDenoisePath(Path[Tensor, Tensor, Tensor]):
    def __init__(self, init_dist: Distribution[Tensor], a: float, b: float):
        self.init_dist = init_dist
        self.kappa = CubicKappa(a, b)
    def sample(self, data1, t):
        k, dk = self.kappa(t)
        data0 = self.init_dist.sample()
        data = data0 * (1-k) + data1 * k
        return data, data1
    def sample_init(self):
        return self.init_dist.sample()
    def update(self, datas, bpred, t, delta_t, alpha):
        bpred = bpred.to(datas[0].device)
        k, dk = self.kappa(t)
        datas = [
            data + (bpred[i]-data)*delta_t*dk/(1-k) for i, data in enumerate(datas)
        ]
        return datas
    def criterion(self, targets, bpred):
        btarget = torch.stack(targets).to(bpred.device)
        loss = F.mse_loss(bpred, btarget)
        return Loss([loss], ['loss'], [1.0])

# MolData
class MolPath[NT, CT, NBP, CBP](Path[MolData, tuple[NT, CT], tuple[NBP, CBP]]):
    def __init__(self, node_path: Path[Tensor, NT, NBP], coord_path: Path[Tensor, CT, CBP], coord_weight: float):
        self.node_path = node_path
        self.coord_path = coord_path
        self.coord_weight = coord_weight
    def sample(self, data1, t):
        node, node_tgt = self.node_path.sample(data1.node, t)
        coord, coord_tgt = self.coord_path.sample(data1.coord, t)
        return MolData(node, coord), (node_tgt, coord_tgt)
    def sample_init(self):
        return MolData(self.node_path.sample_init(), self.coord_path.sample_init())
    def update(self, datas, bpred, t, delta_t, alpha):
        node_bpred, coord_bpred = bpred
        nodes = [data.node for data in datas]
        coords = [data.coord for data in datas]
        nodes = self.node_path.update(nodes, node_bpred, t, delta_t, alpha)
        coords = self.coord_path.update(coords, coord_bpred, t, delta_t, alpha)
        return [MolData(node, coord) for node, coord in zip(nodes, coords)]
    def criterion(self, targets, bpred):
        node_targets, coord_targets = zip(*targets)
        node_loss = self.node_path.criterion(node_targets, bpred[0])
        coord_loss = self.coord_path.criterion(coord_targets, bpred[1])
        node_loss.names = ['node_'+name for name in node_loss.names]
        coord_loss.names = ['coord_'+name for name in coord_loss.names]
        coord_loss.weights = [w*self.coord_weight for w in coord_loss.weights]
        return Loss.cat(node_loss, coord_loss)
