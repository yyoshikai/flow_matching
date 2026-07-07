import yaml
import torch.nn as nn
from argparse import ArgumentParser


class MolFMModel(nn.Module):
    def __init__(self, )


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--studyname', required=True)
    args = parser.parse_args()

    for batch in train_batch_iter:
        streamer.put_data(batch)
        inputs, ts, targets = zip(*batch)
        pred = model(inputs, ts)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        if stop_criterion(model, batch, loss):
            break


