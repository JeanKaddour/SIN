from torch import nn

from models.building_blocks.mlp import MLP


class CovariatesFeatureExtractor(nn.Module):
    def __init__(self, args):
        super(CovariatesFeatureExtractor, self).__init__()
        dim_output = (
            args.dim_output if args.model == "sin" else args.dim_output_covariates
        )
        self.covariates_net = MLP(
            dim_input=args.dim_covariates,
            dim_hidden=args.dim_hidden_covariates,
            dim_output=dim_output,
            num_layers=args.num_covariates_layer,
            batch_norm=args.mlp_batch_norm,
            initialiser=args.initialiser,
            dropout=args.dropout,
            activation=args.activation,
            leaky_relu=args.leaky_relu,
            is_output_activation=args.output_activation_treatment_features,
        )

    def forward(self, unit):
        return self.covariates_net(unit)
