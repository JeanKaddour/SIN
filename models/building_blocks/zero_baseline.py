from torch import Tensor
from torch_geometric.data.batch import Batch

from models.building_blocks.neural_network import NeuralNetworkEstimator


class ZeroBaseline(NeuralNetworkEstimator):

    def __init__(self, args):
        super(ZeroBaseline, self).__init__(args)

    def forward(self, batch: Batch):
        return Tensor(len(batch.covariates) * [0.])
