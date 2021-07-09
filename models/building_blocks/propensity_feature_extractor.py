from torch import nn, Tensor

from models.building_blocks.mlp import MLP


class PropensityNet(nn.Module):
    def __init__(self, args):
        super(PropensityNet, self).__init__()
        self.mlp = MLP(
            dim_input=args.dim_covariates,
            dim_hidden=args.dim_hidden_propensity,
            dim_output=args.dim_output,
            num_layers=args.num_propensity_layers,
            batch_norm=args.mlp_batch_norm,
            initialiser=args.initialiser,
            dropout=args.pro_dropout,
            activation=args.activation,
            leaky_relu=args.leaky_relu,
            is_output_activation=args.output_activation_treatment_features,
        )

    def forward(self, covariates: Tensor):
        return self.mlp(covariates)
