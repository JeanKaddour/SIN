import torch.nn.functional as F
from torch import Tensor, cat
from torch_geometric.data.batch import Batch

from models.building_blocks.covariates_feature_extractor import (
    CovariatesFeatureExtractor,
)
from models.building_blocks.hsic_utils import hsic_normalized
from models.building_blocks.neural_network import NeuralNetworkEstimator
from models.building_blocks.outcome_model import OutcomeModel
from models.building_blocks.treatment_feature_extractor import TreatmentFeatureExtractor
from models.building_blocks.utils import get_optimizer_scheduler


class GraphITE(NeuralNetworkEstimator):
    def __init__(self, args):
        super(GraphITE, self).__init__(args)
        self.treatment_net = TreatmentFeatureExtractor(args=args)
        self.covariates_net = CovariatesFeatureExtractor(args=args)
        self.outcome_net = OutcomeModel(args=args)
        self.independence_regularisation_coeff = args.independence_regularisation_coeff
        self.optimizer, self.lr_scheduler = get_optimizer_scheduler(
            args=args, model=self
        )
        self.is_multi_relational = args.gnn_multirelational

    def loss(self, prediction: Tensor, batch: Batch):
        pred_outcome, unit_features, treatment_features = (
            prediction[0].view(-1),
            prediction[1],
            prediction[2],
        )
        target_outcome = batch.y
        outcome_loss = F.mse_loss(input=pred_outcome, target=target_outcome)
        HSIC_regularisation = self.independence_regularisation_coeff * hsic_normalized(
            x=unit_features, y=treatment_features
        )
        return outcome_loss + HSIC_regularisation

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
        return outcome, covariates_features, treatment_features

    def test_prediction(self, batch: Batch):
        return self.forward(batch)[0].view(-1)
