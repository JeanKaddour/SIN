from typing import Union

import numpy as np

from experiments.utils import sample_uniform_weights
from simulation.outcome_generators import OutcomeGenerator, generate_outcome_sw


class SmallWorldSimulator(OutcomeGenerator):

    def __init__(self, id_to_graph_dict: dict, noise_mean: float = 0., noise_std: float = 1., dim_covariates: int = 20):
        super().__init__(id_to_graph_dict=id_to_graph_dict, noise_mean=noise_mean, noise_std=noise_std)
        self.covariates_weights = sample_uniform_weights(num_weights=3, dim_covariates=dim_covariates)

    def set_id_to_graph_dict(self, id_to_graph_dict: dict):
        self.id_to_graph_dict = id_to_graph_dict

    def generate_outcomes_for_units(self, units: list, treatment_ids: list) -> np.ndarray:
        return self.__generate_outcomes(units=units, treatment_ids=treatment_ids)

    def generate_outcomes_for_unit(self, unit: list, treatment_ids: list) -> np.ndarray:
        units = np.repeat(np.expand_dims(unit, axis=0), len(treatment_ids), axis=0)
        return self.__generate_outcomes(units=units, treatment_ids=treatment_ids)

    def __generate_outcomes(self, units: Union[list, np.ndarray], treatment_ids: list) -> np.ndarray:
        outcomes = []
        for covariates, treatment_id in zip(units, treatment_ids):
            graph_features = self.id_to_graph_dict[treatment_id]['graph_features']
            outcome = generate_outcome_sw(covariates=covariates, graph_features=graph_features,
                                          random_weights=self.covariates_weights) + self._sample_noise()
            outcomes.append(outcome)
        return np.array(outcomes).squeeze()
