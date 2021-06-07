from abc import ABC

import numpy as np


class OutcomeGenerator(ABC):
    def __init__(self, id_to_graph_dict: dict, noise_mean: float = 0., noise_std: float = 1.):
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.id_to_graph_dict = id_to_graph_dict

    def _sample_noise(self):
        return np.random.normal(loc=self.noise_mean, scale=self.noise_std)


def generate_outcome_sw(covariates, graph_features, random_weights):
    baseline_effect = 100. * np.dot(random_weights[0], covariates)
    treatment_effect = .2 * graph_features[0] ** 2 * np.dot(
        random_weights[1], covariates)
    treatment_effect += graph_features[1] * np.dot(
        random_weights[2], covariates)
    return baseline_effect + treatment_effect


def generate_outcome_tcga(unit_features, pca_features, prop, random_weights):
    baseline_effect = 10. * np.dot(random_weights[0], unit_features)
    treatment_effect = np.dot(prop[:8], pca_features) * .01
    return baseline_effect + treatment_effect
