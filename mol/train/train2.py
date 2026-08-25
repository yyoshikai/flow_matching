from torch import Tensor
from egnn_pytorch import EGNN_Network
from src.fm.train import FMModel, train_fm


Mol = tuple[Tensor, Tensor, Tensor] # [node, coord, charge]
MolBPred = tuple[Tensor, Tensor, Tensor]

class MolFMModel(FMModel[Mol, MolBPred]):
    def __init__(self):
        

    def forward(self, datas, ts):
        nodes, coords, charges = zip(*datas)



if __name__ == "__main__":



    train_fm(model, optimizer, data_iter, path, streamer, stop_criterion)

