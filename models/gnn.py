from torch import Tensor
from torch import cat
from torch.nn import functional as F
from torch_geometric.data.batch import Batch

from models.building_blocks.neural_network import NeuralNetworkEstimator
from models.building_blocks.outcome_model import OutcomeModel
from models.building_blocks.treatment_feature_extractor import TreatmentFeatureExtractor
from models.building_blocks.covariates_feature_extractor import (
    CovariatesFeatureExtractor,
)
from models.building_blocks.utils import get_optimizer_scheduler


class GNNRegressionModel(NeuralNetworkEstimator):
    def __init__(self, args):
        super(GNNRegressionModel, self).__init__(args)
        self.treatment_net = TreatmentFeatureExtractor(args=args)
        self.covariates_net = CovariatesFeatureExtractor(args=args)
        self.outcome_net = OutcomeModel(args=args)
        self.optimizer, self.lr_scheduler = get_optimizer_scheduler(
            args=args, model=self
        )
        self.is_multi_relational = args.gnn_multirelational

    def loss(self, prediction: Tensor, batch: Batch):
        target_outcome = batch.y
        outcome_loss = F.mse_loss(input=prediction.view(-1), target=target_outcome)
        return outcome_loss

    def forward(self, batch: Batch):
        treatment_node_features, treatment_edges, covariates, batch_assignments = (
            batch.x,
            batch.edge_index,
            batch.covariates,
            batch.batch,
        )
        treatment_edge_types = batch.edge_types if self.is_multi_relational else None
        treatment_features = self.treatment_net(
            treatment_node_features,
            treatment_edges,
            treatment_edge_types,
            batch_assignments,
        )
        covariates_features = self.covariates_net(covariates)
        outcome_net_input = cat([treatment_features, covariates_features], dim=1)
        outcome = self.outcome_net(outcome_net_input)
        return outcome

    def forward_treatment_net(self, batch: Batch):
        treatment_node_features, treatment_edges, batch_assignments = (
            batch.x,
            batch.edge_index,
            batch.batch,
        )
        treatment_edge_types = batch.edge_types if self.is_multi_relational else None
        treatment_features = self.treatment_net(
            treatment_node_features,
            treatment_edges,
            treatment_edge_types,
            batch_assignments,
        )
        return treatment_features
