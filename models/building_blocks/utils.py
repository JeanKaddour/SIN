from argparse import Namespace
from typing import Union, Callable, Optional, Iterator

from torch import nn, optim
from torch_geometric.nn import GCNConv, GATConv, GraphConv, RGCNConv


def get_activation(name: str, leaky_relu: Optional[float] = 0.5) -> nn.Module:
    if name == "leaky_relu":
        return nn.LeakyReLU(leaky_relu)
    elif name == "rrelu":
        return nn.RReLU()
    elif name == "relu":
        return nn.ReLU()
    elif name == "elu":
        return nn.ELU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name == "selu":
        return nn.SELU()
    else:
        raise Exception("Unknown activation")


def get_gnn_conv(name: str) -> Union[GCNConv, GATConv, GraphConv, RGCNConv]:
    if name == "gcn":
        return GCNConv
    elif name == "gat":
        return GATConv
    elif name == "graph_conv":
        return GraphConv
    elif name == "rcgn":
        return RGCNConv
    else:
        raise Exception("Unknown GNN layer")


def get_initialiser(name: str) -> Callable:
    if name == "orthogonal":
        return nn.init.orthogonal_
    elif name == "xavier":
        return nn.init.xavier_uniform_
    elif name == "kaiming":
        return nn.init.kaiming_uniform_
    elif name == "none":
        pass
    else:
        raise Exception("Unknown init method")


def get_optimizer(
    args: Namespace, params: Iterator[nn.Parameter], net: Optional[str] = None
) -> optim.Optimizer:
    weight_decay = args.weight_decay
    lr = args.lr
    if net == "propensity":
        weight_decay = args.pro_weight_decay
        lr = args.pro_lr
    elif net == "como":
        lr = args.como_lr
        weight_decay = args.como_weight_decay
    elif net == "gnn":
        lr = args.como_lr
        weight_decay = args.gnn_weight_decay

    optimizer = None
    if args.optimizer == "sgd":
        optimizer = optim.SGD(params, lr=lr, weight_decay=weight_decay)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif args.optimizer == "amsgrad":
        optimizer = optim.Adam(params, lr=lr, amsgrad=True, weight_decay=weight_decay)
    return optimizer


class NoneScheduler:
    def step(self):
        pass


def get_lr_scheduler(
    args: Namespace, optimizer: optim.Optimizer
) -> Union[
    optim.lr_scheduler.ExponentialLR,
    optim.lr_scheduler.CosineAnnealingLR,
    optim.lr_scheduler.CyclicLR,
    NoneScheduler,
]:
    if args.lr_scheduler == "exponential":
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=0)
    elif args.lr_scheduler == "cycle":
        return optim.lr_scheduler.CyclicLR(
            optimizer, 0, max_lr=args.lr, step_size_up=20, cycle_momentum=False
        )
    elif args.lr_scheduler == "none":
        return NoneScheduler()


def get_optimizer_scheduler(
    args: Namespace, model: nn.Module, net: Optional[str] = None
):
    optimizer = get_optimizer(args=args, params=model.parameters(), net=net)
    lr_scheduler = get_lr_scheduler(args=args, optimizer=optimizer)
    return optimizer, lr_scheduler
