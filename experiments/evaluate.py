import argparse
import logging
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch import nn, no_grad
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from data.dataset import TestUnit, TestUnits, create_pt_geometric_dataset
from data.utils import get_treatment_graphs


def valid_evaluation(
    model: nn.Module,
    device: torch.device,
    val_loader: DataLoader,
    epoch: int,
    val_loss_name: str,
) -> float:
    """Computes the validation loss during training for hyper-parameter tuning and early stopping."""
    model.eval()
    val_error = 0.0
    for batch_idx, batch in enumerate(val_loader):
        batch = batch.to(device)
        target_outcome = batch.y
        prediction = model.test_prediction(batch)
        val_error += F.mse_loss(
            input=prediction, target=target_outcome, reduction="sum"
        ).item()
    val_error /= len(val_loader.dataset)
    logging.info(f"{val_loss_name}: {val_error:.6f}")
    wandb.log({"epoch": epoch, val_loss_name: val_error})
    return val_error


def predict_outcomes(
    model: nn.Module,
    device: torch.device,
    test_units: List[TestUnit],
    id_to_graph_dict: dict,
    unit_description: str,
) -> None:
    """Predicts and stores (pseudo-)outcomes for model evaluation."""
    model.eval()
    logging.info(f"Inference on {unit_description} units...")
    for i, test_unit in enumerate(tqdm(test_units)):
        treatment_ids = test_unit.get_treatment_ids()
        treatment_graphs = get_treatment_graphs(
            treatment_ids=treatment_ids, id_to_graph_dict=id_to_graph_dict
        )
        unit = test_unit.get_covariates()
        units = np.repeat(np.expand_dims(unit, axis=0), len(treatment_ids), axis=0)
        test_unit_pt_dataset = create_pt_geometric_dataset(
            units=units, treatment_graphs=treatment_graphs
        )
        with no_grad():
            batch = Batch.from_data_list(test_unit_pt_dataset).to(device)
            predicted_outcomes = model.test_prediction(batch).cpu().numpy()
        predicted_outcomes_dict = dict(zip(treatment_ids, predicted_outcomes))
        test_unit.set_predicted_outcomes(predicted_outcomes=predicted_outcomes_dict)


def test_evaluation(
    model: nn.Module,
    device: torch.device,
    test_dataset: TestUnits,
    args: argparse.Namespace,
) -> dict:
    """Evaluates model on test data and returns test errors for varying values of K."""
    id_to_graph_dict = test_dataset.get_id_to_graph_dict()
    predict_outcomes(
        model=model,
        device=device,
        test_units=test_dataset.get_test_units(in_sample=True),
        id_to_graph_dict=id_to_graph_dict,
        unit_description="in-sample",
    )
    predict_outcomes(
        model=model,
        device=device,
        test_units=test_dataset.get_test_units(in_sample=False),
        id_to_graph_dict=id_to_graph_dict,
        unit_description="out-sample",
    )
    test_errors = {}
    for k in range(args.min_test_assignments, args.max_test_assignments + 1):
        logging.info(f"Evaluate errors for K={k}...")
        test_errors[k] = {"in_sample": {}, "out_sample": {}}
        unweighted_errors_in_sample, weighted_errors_in_sample = [], []
        for test_unit in tqdm(test_dataset.get_test_units(in_sample=True)):
            unweighted_error, weighted_error = test_unit.evaluate_predictions(k=k)
            unweighted_errors_in_sample.append(unweighted_error)
            weighted_errors_in_sample.append(weighted_error)
        test_errors[k]["in_sample"]["unweighted"] = np.mean(unweighted_errors_in_sample)
        test_errors[k]["in_sample"]["weighted"] = np.mean(weighted_errors_in_sample)
        unweighted_errors_out_sample, weighted_errors_out_sample = [], []
        for test_unit in test_dataset.get_test_units(in_sample=False):
            unweighted_error, weighted_error = test_unit.evaluate_predictions(k=k)
            unweighted_errors_out_sample.append(unweighted_error)
            weighted_errors_out_sample.append(weighted_error)
        test_errors[k]["out_sample"]["unweighted"] = np.mean(
            unweighted_errors_out_sample
        )
        test_errors[k]["out_sample"]["weighted"] = np.mean(weighted_errors_out_sample)

        logging.info(
            f'K={k}: \t In Sample: {test_errors[k]["in_sample"]}, \t Out Sample : {test_errors[k]["out_sample"]}'
        )

    wandb.log({"all_test_errors": test_errors})
    return test_errors
