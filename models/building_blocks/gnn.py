from torch import Tensor, nn
from torch_geometric.nn import BatchNorm, global_mean_pool

from models.building_blocks.utils import get_activation, get_gnn_conv, get_initialiser


def create_batch_norm_gnn_layers(num_layers: int, dim_hidden: int):
    batch_norm_layers = nn.ModuleList()
    for i in range(num_layers):
        batch_norm_layers.append(BatchNorm(in_channels=dim_hidden))
    return batch_norm_layers


def create_gnn_layers(
    gnn_conv: str,
    num_layers: int,
    dim_input: int,
    dim_hidden: int,
    dim_output: int,
    num_relations: int,
    num_bases: int,
):
    conv_operator = get_gnn_conv(name=gnn_conv)
    gnn_layers = nn.ModuleList()
    if gnn_conv == "rcgn":
        if num_bases == -1:
            num_bases = None
        # Input layer
        gnn_layers.append(
            conv_operator(
                in_channels=dim_input,
                out_channels=dim_hidden,
                num_relations=num_relations,
                num_bases=num_bases,
            )
        )
        # Hidden layers
        for i in range(1, num_layers):
            gnn_layers.append(
                conv_operator(
                    in_channels=dim_hidden,
                    out_channels=dim_hidden,
                    num_relations=num_relations,
                    num_bases=num_bases,
                )
            )
    else:
        # Input layer
        gnn_layers.append(conv_operator(in_channels=dim_input, out_channels=dim_hidden))
        # Hidden layers
        for i in range(1, num_layers):
            gnn_layers.append(
                conv_operator(in_channels=dim_hidden, out_channels=dim_hidden)
            )
    # Output layer
    gnn_layers.append(nn.Linear(dim_hidden, dim_output))
    return gnn_layers


def init_layers(initialiser_name: str, layers: nn.ModuleList):
    initialiser = get_initialiser(initialiser_name)
    for layer in layers:
        initialiser(layer.weight)


class GNN(nn.Module):
    def __init__(
        self,
        gnn_conv: str,
        dim_input: int,
        dim_hidden: int,
        dim_output: int,
        num_layers: int,
        batch_norm: bool,
        initialiser: str,
        dropout: float,
        activation: str,
        leaky_relu: float,
        is_output_activation: bool,
        num_relations: int,
        num_bases: int,
        is_multi_relational: bool,
    ):
        super().__init__()
        self.layers = create_gnn_layers(
            gnn_conv,
            num_layers=num_layers,
            dim_input=dim_input,
            dim_hidden=dim_hidden,
            dim_output=dim_output,
            num_relations=num_relations,
            num_bases=num_bases,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        self.batch_norm_layers = (
            create_batch_norm_gnn_layers(num_layers=num_layers, dim_hidden=dim_hidden)
            if batch_norm
            else None
        )
        self.activation_function = get_activation(
            name=activation, leaky_relu=leaky_relu
        )
        self.is_output_activation = is_output_activation
        self.is_multi_relational = is_multi_relational

    def forward_single_edge_type(
        self, nodes: Tensor, edges: Tensor, batch_assignments: Tensor
    ):
        for i in range(len(self.layers) - 1):
            nodes = self.layers[i](nodes, edges)
            nodes = self.activation_function(nodes)
            if self.batch_norm_layers:
                nodes = self.batch_norm_layers[i](nodes)
        nodes = global_mean_pool(nodes, batch=batch_assignments)
        if self.dropout:
            nodes = self.dropout(nodes)
        nodes = self.layers[-1](nodes)
        if self.is_output_activation:
            nodes = self.activation_function(nodes)
        return nodes

    def forward(
        self,
        nodes: Tensor,
        edges: Tensor,
        edge_types: Tensor,
        batch_assignments: Tensor,
    ):
        if self.is_multi_relational:
            return self.forward_multirelational(
                nodes=nodes,
                edges=edges,
                edge_types=edge_types,
                batch_assignments=batch_assignments,
            )
        else:
            return self.forward_single_edge_type(
                nodes=nodes, edges=edges, batch_assignments=batch_assignments
            )

    def forward_multirelational(
        self,
        nodes: Tensor,
        edges: Tensor,
        batch_assignments: Tensor,
        edge_types: Tensor,
    ):
        for i in range(len(self.layers) - 1):
            nodes = self.layers[i](nodes, edges, edge_types)
            nodes = self.activation_function(nodes)
            if self.batch_norm_layers:
                nodes = self.batch_norm_layers[i](nodes)
        nodes = global_mean_pool(nodes, batch=batch_assignments)
        nodes = self.layers[-1](nodes)
        if self.is_output_activation:
            nodes = self.activation_function(nodes)
        if self.dropout:
            nodes = self.dropout(nodes)
        return nodes
