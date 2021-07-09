from typing import Union

import numpy as np

from experiments.utils import sample_uniform_weights
from simulation.outcome_generators import OutcomeGenerator, generate_outcome_tcga


class TCGASimulator(OutcomeGenerator):
    def __init__(
        self,
        id_to_graph_dict: dict,
        noise_mean: float = 0.0,
        noise_std: float = 1.0,
        dim_covariates: int = 25,
    ):
        super().__init__(
            id_to_graph_dict=id_to_graph_dict,
            noise_mean=noise_mean,
            noise_std=noise_std,
        )
        self.covariates_weights = sample_uniform_weights(
            num_weights=3, dim_covariates=dim_covariates
        )

    def set_id_to_graph_dict(self, id_to_graph_dict: dict) -> None:
        self.id_to_graph_dict = id_to_graph_dict

    def generate_outcomes_for_units(
        self, pca_features: list, unit_features: list, treatment_ids: list
    ) -> np.ndarray:
        return self.__generate_outcomes(
            pca_features=pca_features,
            unit_features=unit_features,
            treatment_ids=treatment_ids,
        )

    def generate_outcomes_for_unit(
        self, pca_features, unit_features, treatment_ids
    ) -> np.ndarray:
        pca_features = np.repeat(
            np.expand_dims(pca_features, axis=0), len(treatment_ids), axis=0
        )
        unit_features = np.repeat(
            np.expand_dims(unit_features, axis=0), len(treatment_ids), axis=0
        )
        return self.__generate_outcomes(
            pca_features=pca_features,
            unit_features=unit_features,
            treatment_ids=treatment_ids,
        )

    def __generate_outcomes(
        self,
        pca_features: Union[list, np.ndarray],
        unit_features: Union[list, np.ndarray],
        treatment_ids: list,
    ) -> np.ndarray:
        outcomes = []
        for pca_features, unit_features, treatment_id in zip(
            pca_features, unit_features, treatment_ids
        ):
            prop = self.id_to_graph_dict[treatment_id]["prop"]
            outcome = (
                generate_outcome_tcga(
                    unit_features=unit_features,
                    pca_features=pca_features,
                    prop=prop,
                    random_weights=self.covariates_weights,
                )
                + self._sample_noise()
            )
            outcomes.append(outcome)
        return np.array(outcomes).squeeze()
