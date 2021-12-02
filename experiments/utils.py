import argparse
import json
import random
from argparse import Namespace
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch as th
import yaml
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch

import wandb
from data.dataset import (TestUnits, create_pt_geometric_dataset,
                          create_pt_geometric_dataset_only_graphs)
from data.utils import split_train_val
from experiments.io import load_train_dataset, pickle_dump
from models.building_blocks.zero_baseline import ZeroBaseline
from models.cat import CategoricalTreatmentRegressionModel
from models.gnn import GNNRegressionModel
from models.graphite import GraphITE
from models.sin import SIN


def save_args(args: argparse.Namespace, path: str) -> None:
    with open(path, "w") as f:
        json.dump(args.__dict__, f, indent=2)


def get_model(args: Namespace, device) -> th.nn.Module:
    str_to_model_dict = {
        "gnn": GNNRegressionModel,
        "graphite": GraphITE,
        "zero": ZeroBaseline,
        "sin": SIN,
        "cat": CategoricalTreatmentRegressionModel,
    }
    model = str_to_model_dict[args.model](args=args).to(device)
    wandb.watch(model, log="all", log_freq=args.log_interval)

    return model


def init_seeds(seed: int) -> None:
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    th.cuda.manual_seed_all(seed)


def sample_uniform_weights(
    num_weights: int, dim_covariates: int, low: float = 0.0, high: float = 1.0
) -> np.ndarray:
    weights = np.zeros(shape=(num_weights, dim_covariates))
    for i in range(num_weights):
        weights[i] = np.random.uniform(low=low, high=high, size=(dim_covariates))
        weights[i] /= np.linalg.norm(weights[i])
    return weights


def compute_graph_embeddings(
    model: th.nn.Module, device, treatment_ids: list, id_to_graph_dict: dict
) -> th.Tensor:
    graphs = [id_to_graph_dict[id] for id in treatment_ids]
    graph_data = Batch.from_data_list(create_pt_geometric_dataset_only_graphs(graphs))
    with th.no_grad():
        graph_data = graph_data.to(device)
        graph_embeddings = model.forward_treatment_net(graph_data).cpu()
    return graph_embeddings


def get_ids_with_closest_distance(
    target_embeddings: th.Tensor, source_embeddings: th.Tensor, source_ids: list
) -> list:
    closest_graph_ids = []
    pairwise_distances = th.cdist(target_embeddings, source_embeddings)
    for i in range(pairwise_distances.shape[0]):
        row = pairwise_distances[i]
        closest_idx = th.argmin(row)
        closest_graph_id = source_ids[closest_idx]
        closest_graph_ids.append(closest_graph_id)
    return closest_graph_ids


def save_run_results(
    test_units_with_predictions: TestUnits,
    test_errors: dict,
    time_str: str,
    args: Namespace,
) -> None:
    custom_results_path = (
        args.results_path + f"{args.task}/{args.seed}/{args.model}/{time_str}/"
    )
    file_path_test_units = custom_results_path + "test_units.p"
    file_path_test_errors = custom_results_path + "test_errors.p"
    file_path_args = custom_results_path + "args.p"
    Path(custom_results_path).mkdir(parents=True, exist_ok=True)
    pickle_dump(file_name=file_path_test_units, content=test_units_with_predictions)
    pickle_dump(file_name=file_path_test_errors, content=test_errors)
    pickle_dump(file_name=file_path_args, content=args)


def read_yaml(path: str) -> dict:
    with open(path, "r") as stream:
        data_loaded = yaml.safe_load(stream)
    return data_loaded


def get_train_and_val_dataset(
    args: Namespace,
) -> Tuple[List[Data], List[Data]]:
    in_sample_data = load_train_dataset(args=args)
    units = (
        in_sample_data.get_units()["features"]
        if args.task in ["tcga"]
        else in_sample_data.get_units()
    )
    graphs = in_sample_data.get_treatment_graphs()
    outcomes = in_sample_data.get_outcomes()
    train_data, val_data = split_train_val(
        units=units, graphs=graphs, outcomes=outcomes, args=args
    )
    train_data_pt = create_pt_geometric_dataset(
        units=train_data["units"],
        treatment_graphs=train_data["graphs"],
        outcomes=train_data["outcomes"],
    )
    val_data_pt = create_pt_geometric_dataset(
        units=val_data["units"],
        treatment_graphs=val_data["graphs"],
        outcomes=val_data["outcomes"],
    )

    assert train_data is not None
    assert val_data_pt is not None
    return train_data_pt, val_data_pt


def get_train_and_val_pt_datasets(
    units: list, graphs: list, outcomes: list, args: Namespace
) -> Tuple[List[Data], List[Data]]:
    train_data, val_data = split_train_val(
        units=units, graphs=graphs, outcomes=outcomes, args=args
    )
    train_data_pt = create_pt_geometric_dataset(
        units=train_data["units"],
        treatment_graphs=train_data["graphs"],
        outcomes=train_data["outcomes"],
    )
    val_data_pt = create_pt_geometric_dataset(
        units=val_data["units"],
        treatment_graphs=val_data["graphs"],
        outcomes=val_data["outcomes"],
    )
    return train_data_pt, val_data_pt
