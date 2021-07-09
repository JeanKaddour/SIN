from torch import nn, Tensor

from models.building_blocks.gnn import GNN


class TreatmentFeatureExtractor(nn.Module):
    def __init__(self, args):
        super(TreatmentFeatureExtractor, self).__init__()
        dim_output = (
            args.dim_output if args.model == "gin" else args.dim_output_covariates
        )
        self.treatment_net = GNN(
            gnn_conv=args.gnn_conv,
            dim_input=args.dim_node_features,
            dim_hidden=args.dim_hidden_treatment,
            dim_output=dim_output,
            num_layers=args.num_treatment_layer,
            batch_norm=args.gnn_batch_norm,
            initialiser=args.initialiser,
            dropout=args.gnn_dropout,
            activation=args.activation,
            leaky_relu=args.leaky_relu,
            is_output_activation=args.output_activation_treatment_features,
            num_relations=args.gnn_num_relations,
            num_bases=args.gnn_num_bases,
            is_multi_relational=args.gnn_multirelational,
        )

    def forward(
        self,
        treatment_node_features: Tensor,
        treatment_edges: Tensor,
        edge_types: Tensor,
        batch_assignments: Tensor,
    ):
        return self.treatment_net.forward(
            nodes=treatment_node_features,
            edges=treatment_edges,
            edge_types=edge_types,
            batch_assignments=batch_assignments,
        )
