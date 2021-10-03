from typing import List, Tuple, Union

import numpy as np
import torch
from torch_geometric.data import Data

from data.utils import get_treatment_combinations, get_treatment_graphs


class Dataset:
    """
    This class maintains the training data.

    It includes
    * observations in the form (x_i, t_i, y_i)
    * meta-data about the observations
    * treatment graphs
    """

    def __init__(self, data_dict: dict):
        assert data_dict
        self.data_dict = data_dict
        assert "units" in data_dict

    def add_assigned_treatments(self, treatment_ids: list) -> None:
        assert len(treatment_ids) > 0, "List of treatment IDs can not be empty"
        self.data_dict["treatment_ids"] = treatment_ids

    def add_outcomes(self, outcomes: np.ndarray) -> None:
        assert len(outcomes) > 0, "Outcomes NumPy array can not be empty"
        self.data_dict["outcomes"] = outcomes

    def get_num_units(self) -> int:
        return len(self.data_dict["units"])

    def get_dim_covariates(self) -> int:
        return self.data_dict["dim_covariates"]

    def get_units(self) -> Union[list, np.ndarray]:
        return self.data_dict["units"]

    def get_all_treatments(self) -> list:
        return self.data_dict["all_treatments"]

    def get_data_dict(self) -> dict:
        return self.data_dict

    def get_treatment_graphs(self) -> list:
        return get_treatment_graphs(
            treatment_ids=self.get_assigned_treatment_ids(),
            id_to_graph_dict=self.get_id_to_graph_dict(),
        )

    def get_assigned_treatment_ids(self) -> list:
        assert "treatment_ids" in self.data_dict
        return self.data_dict["treatment_ids"]

    def get_unique_treatment_ids(self) -> list:
        assert "treatment_ids" in self.data_dict
        return np.unique(self.get_assigned_treatment_ids()).tolist()

    def get_id_to_graph_dict(self) -> dict:
        return self.data_dict["id_to_graph_dict"]

    def get_outcomes(self) -> Union[None, list]:
        if self.has_outcomes():
            return self.data_dict["outcomes"]
        return None

    def has_outcomes(self) -> bool:
        return "outcomes" in self.data_dict

    def set_id_to_graph_dict(self, id_to_graph_dict: dict) -> None:
        self.data_dict["id_to_graph_dict"] = id_to_graph_dict


class TestUnit:
    """This class maintains one covariate vector with multiple treatment-outcome pairs for model evaluation."""

    def __init__(
        self, covariates, treatment_ids, treatment_propensities, true_outcomes
    ):
        self.data_dict = {}
        self.add_covariates(covariates=covariates)
        self.add_treatments(
            treatment_ids=treatment_ids, treatment_propensities=treatment_propensities
        )
        self.set_true_outcomes(true_outcomes=true_outcomes)

    def add_covariates(self, covariates) -> None:
        self.data_dict["covariates"] = covariates

    def add_treatments(self, treatment_ids, treatment_propensities) -> None:
        assert len(treatment_ids) > 0, "List of treatment IDs can not be empty"
        assert (
            len(treatment_propensities) > 0
        ), "List of treatment propensities can not be empty"
        self.data_dict["treatment_ids"] = treatment_ids
        self.data_dict["treatment_ids_to_propensity_dict"] = dict(
            zip(self.data_dict["treatment_ids"], treatment_propensities)
        )
        self.data_dict["propensity_to_treatment_ids_dict"] = dict(
            zip(treatment_propensities, self.data_dict["treatment_ids"])
        )
        self.data_dict["propensities"] = sorted(treatment_propensities, reverse=True)

    def evaluate_predictions(self, k: int) -> Tuple[np.ndarray, float]:
        true_causal_effects, weights = self.get_true_causal_effects(k=k)
        predicted_causal_effects = self.get_predicted_causal_effects(k=k)
        squared_error = np.square(true_causal_effects - predicted_causal_effects)
        weighted_squared_error = np.average(squared_error, weights=weights)
        return np.mean(squared_error), weighted_squared_error

    def get_covariates(self) -> list:
        return self.data_dict["covariates"]

    def get_k_likely_treatment_ids(self, k: int) -> list:
        return [
            self.data_dict["propensity_to_treatment_ids_dict"][propensity]
            for propensity in self.data_dict["propensities"][:k]
        ]

    def get_treatment_ids(self) -> list:
        return self.data_dict["treatment_ids"]

    def get_treatment_combinations(self, k: int) -> list:
        k_likely_treatment_ids = self.get_k_likely_treatment_ids(k)
        return get_treatment_combinations(treatment_ids=k_likely_treatment_ids)

    def get_predicted_causal_effects(self, k: int) -> np.ndarray:
        predicted_causal_effects = []
        for combination in self.get_treatment_combinations(k):
            treatment_1_id, treatment_2_id = combination[0], combination[1]
            outcome_1, outcome_2 = (
                self.data_dict["predicted_outcomes"][treatment_1_id],
                self.data_dict["predicted_outcomes"][treatment_2_id],
            )

            predicted_causal_effects.append(outcome_1 - outcome_2)
        return np.array(predicted_causal_effects)

    def get_true_causal_effects(self, k: int) -> Tuple[np.ndarray, list]:
        true_causal_effects = []
        weights = []
        for combination in self.get_treatment_combinations(k):
            treatment_1_id, treatment_2_id = combination[0], combination[1]
            outcome_1, outcome_2 = (
                self.data_dict["true_outcomes"][treatment_1_id],
                self.data_dict["true_outcomes"][treatment_2_id],
            )
            propensity_1, propensity_2 = (
                self.data_dict["treatment_ids_to_propensity_dict"][treatment_1_id],
                self.data_dict["treatment_ids_to_propensity_dict"][treatment_2_id],
            )
            true_causal_effects.append(outcome_1 - outcome_2)
            weights.append(propensity_1 * propensity_2)
        return np.array(true_causal_effects), weights

    def set_predicted_outcomes(self, predicted_outcomes: dict) -> None:
        self.data_dict["predicted_outcomes"] = predicted_outcomes

    def set_true_outcomes(self, true_outcomes: dict) -> None:
        self.data_dict["true_outcomes"] = dict(
            zip(self.get_treatment_ids(), true_outcomes)
        )


class TestUnits:
    """This class represents the test dataset by holding multiple TestUnit objects."""

    def __init__(
        self, test_units_dict: dict, id_to_graph_dict: dict, unseen_treatment_ids: list
    ):
        self.test_units_dict = test_units_dict
        self.id_to_graph_dict = id_to_graph_dict
        self.unseen_treatment_ids = unseen_treatment_ids

    def get_id_to_graph_dict(self) -> dict:
        return self.id_to_graph_dict

    def get_test_units(self, in_sample: bool) -> list:
        return (
            self.test_units_dict["in_sample"]
            if in_sample
            else self.test_units_dict["out_sample"]
        )

    def set_id_to_graph_dict(self, id_to_graph_dict: dict):
        self.id_to_graph_dict = id_to_graph_dict

    def get_unseen_treatment_ids(self) -> list:
        return self.unseen_treatment_ids


class GraphData(Data):
    def __init__(
        self,
        x=None,
        edge_index=None,
        edge_attr=None,
        y=None,
        pos=None,
        normal=None,
        face=None,
        **kwargs
    ):
        super().__init__(x, edge_index, edge_attr, y, pos, normal, face, **kwargs)

    def __cat_dim__(self, key, item):
        if key in ["covariates", "one_hot_encoding"]:
            return None
        else:
            return super().__cat_dim__(key, item)


def create_pt_geometric_dataset_only_graphs(treatment_graphs: list) -> List[GraphData]:
    data_list = []
    is_multi_relational = "edge_types" in treatment_graphs[0]
    for i in range(len(treatment_graphs)):
        c_size, features, edge_index, one_hot_encoding = (
            treatment_graphs[i]["c_size"],
            treatment_graphs[i]["node_features"],
            treatment_graphs[i]["edges"],
            treatment_graphs[i]["one_hot_encoding"],
        )
        edge_index = torch.LongTensor(edge_index)
        if len(edge_index.shape) == 2:
            edge_index = edge_index.transpose(1, 0)
        graph_data = GraphData(x=torch.Tensor(features), edge_index=edge_index)
        if is_multi_relational:
            graph_data.edge_types = torch.LongTensor(
                [treatment_graphs[i]["edge_types"]]
            ).squeeze()
        graph_data.__setitem__("c_size", torch.IntTensor([c_size]))
        data_list.append(graph_data)
    return data_list


def create_pt_geometric_dataset(
    units, treatment_graphs: list, outcomes=None
) -> List[GraphData]:
    unit_tensor = torch.FloatTensor(units)
    data_list = []
    is_multi_relational = "edge_types" in treatment_graphs[0]
    for i in range(len(treatment_graphs)):
        c_size, features, edge_index, one_hot_encoding = (
            treatment_graphs[i]["c_size"],
            treatment_graphs[i]["node_features"],
            treatment_graphs[i]["edges"],
            treatment_graphs[i]["one_hot_encoding"],
        )
        one_hot_encoding = torch.FloatTensor(one_hot_encoding)
        edge_index = torch.LongTensor(edge_index)
        if len(edge_index.shape) == 2:
            edge_index = edge_index.transpose(1, 0)
        graph_data = GraphData(
            x=torch.Tensor(features),
            edge_index=edge_index,
            covariates=unit_tensor[i],
            one_hot_encoding=one_hot_encoding,
        )
        if outcomes is not None:
            graph_data.y = torch.Tensor([outcomes[i]])
        if is_multi_relational:
            graph_data.edge_types = torch.LongTensor(
                [treatment_graphs[i]["edge_types"]]
            ).squeeze()
        graph_data.__setitem__("c_size", torch.IntTensor([c_size]))
        data_list.append(graph_data)
    return data_list
