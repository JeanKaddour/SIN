from torch import Tensor
from torch.nn import functional as F
from torch_geometric.data.batch import Batch

from models.building_blocks.mlp import MLP
from models.building_blocks.neural_network import NeuralNetworkEstimator
from models.building_blocks.utils import get_optimizer_scheduler

"""
COnditional Mean Outcome (COMO) Model 
m(X) := E[Y|X]


"""


class COMONet(NeuralNetworkEstimator):
    def __init__(self, args):
        super(COMONet, self).__init__(args)
        self.mlp = MLP(
            dim_input=args.dim_covariates,
            dim_hidden=args.dim_hidden_como,
            dim_output=1,
            num_layers=args.num_como_layers,
            batch_norm=args.mlp_batch_norm,
            initialiser=args.initialiser,
            dropout=args.como_dropout,
            activation=args.activation,
            leaky_relu=args.leaky_relu,
            is_output_activation=False,
        )
        self.optimizer, self.lr_scheduler = get_optimizer_scheduler(
            args=args, model=self
        )

    def forward_unit(self, covariates: Tensor):
        return self.mlp(covariates)

    def forward(self, batch: Batch):
        return self.forward_unit(batch.covariates)

    def loss(self, prediction: Tensor, batch: Batch):
        target_outcome = batch.y
        outcome_loss = F.mse_loss(input=prediction.view(-1), target=target_outcome)
        return outcome_loss
