import itertools
from operator import itemgetter

import networkx as nx
import numpy as np


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def graph_to_edges(graph: nx.Graph):
    return [(src, dst) for src, dst in graph.edges()]


def normalize_data(x):
    x = (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0))
    return x


def split_train_val(units, graphs, outcomes, args):
    num_train = len(units)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(args.val_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    train_data, val_data = {}, {}
    train_data["units"], train_data["graphs"], train_data["outcomes"] = (
        units[train_idx],
        itemgetter(*train_idx)(graphs),
        outcomes[train_idx],
    )
    val_data["units"], val_data["graphs"], val_data["outcomes"] = (
        units[valid_idx],
        itemgetter(*valid_idx)(graphs),
        outcomes[valid_idx],
    )
    return train_data, val_data


def get_treatment_graphs(treatment_ids: list, id_to_graph_dict: dict):
    return [id_to_graph_dict[i] for i in treatment_ids]


def get_treatment_one_hot_encodings(
    treatment_ids: list, id_to_one_hot_encoding_dict: dict
):
    return [id_to_one_hot_encoding_dict[i] for i in treatment_ids]


def get_treatment_combinations(treatment_ids: list):
    return list(itertools.combinations(treatment_ids, 2))
