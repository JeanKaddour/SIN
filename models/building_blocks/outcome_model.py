from torch import nn, Tensor

from models.building_blocks.mlp import MLP


class OutcomeModel(nn.Module):
    def __init__(self, args):
        super(OutcomeModel, self).__init__()
        self.outcome_net = MLP(dim_input=args.dim_output_treatment + args.dim_output_covariates,
                               dim_hidden=args.dim_hidden_covariates,
                               dim_output=1,
                               num_layers=args.num_final_ff_layer, batch_norm=args.mlp_batch_norm,
                               initialiser=args.initialiser, dropout=args.dropout, activation=args.activation,
                               leaky_relu=args.leaky_relu, is_output_activation=False)

    def forward(self, treatment_and_unit_features: Tensor):
        return self.outcome_net(treatment_and_unit_features)
