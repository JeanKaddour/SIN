from torch import Tensor, cat
from torch.nn import functional as F
from models.building_blocks.covariates_feature_extractor import CovariatesFeatureExtractor
from models.building_blocks.outcome_model import OutcomeModel
from models.building_blocks.neural_network import NeuralNetworkEstimator
from models.building_blocks.utils import get_optimizer_scheduler
from models.building_blocks.mlp import MLP


class CategoricalTreatmentRegressionModel(NeuralNetworkEstimator):

    def __init__(self, args):
        super(CategoricalTreatmentRegressionModel, self).__init__(args)
        self.treatment_net = MLP(dim_input=args.num_treatments, dim_hidden=args.dim_hidden_treatment,
                                 dim_output=args.dim_output_treatment,
                                 num_layers=args.num_treatment_layer, batch_norm=args.mlp_batch_norm,
                                 initialiser=args.initialiser, dropout=args.dropout, activation=args.activation,
                                 leaky_relu=args.leaky_relu, is_output_activation=True)
        self.covariates_net = CovariatesFeatureExtractor(args=args)
        self.outcome_net = OutcomeModel(args=args)
        self.optimizer, self.lr_scheduler = get_optimizer_scheduler(args=args, model=self)

    def loss(self, prediction: Tensor, batch):
        target_outcome = batch.y
        outcome_loss = F.mse_loss(input=prediction.view(-1), target=target_outcome)
        return outcome_loss

    def forward(self, batch):
        covariates, treatment_one_hot_encoding = batch.covariates, batch.one_hot_encoding
        treatment_features = self.treatment_net(treatment_one_hot_encoding)
        covariates_features = self.covariates_net(covariates)
        outcome_net_input = cat([treatment_features, covariates_features], dim=1)
        outcome = self.outcome_net(outcome_net_input)
        return outcome

