from abc import ABC

import numpy as np


def create_treatment_assignment_dict(all_treatment_ids, sorted_selected_idx, propensities_of_selected_treatments):
    selected_treatment_ids = [all_treatment_ids[id] for id in sorted_selected_idx]
    return {'treatment_ids': selected_treatment_ids, 'propensities': propensities_of_selected_treatments}


class TreatmentAssignmentPolicy(ABC):

    def __init__(self, treatment_ids: list, args):
        self.treatment_ids = treatment_ids
        self.bias = args.bias
        self.args = args

    def assign_treatment(self, unit):
        pass

    def get_assignments_for_unit(self, unit, num_test_treatments_per_unit: int = 5):
        pass

    def __get_most_likely_assignments_for_unit(self, unit, num_test_treatments_per_unit: int = 5):
        pass


class RandomTAP(TreatmentAssignmentPolicy):

    def __init__(self, treatment_ids: list, args, weights=None):
        super().__init__(treatment_ids, args)
        self.dim_covariates = args.dim_covariates
        self.policy = args.propensity_covariates_preprocessing
        if weights:
            self.weights = weights
        else:
            self._init_weights()

    def _init_weights_uniform(self):
        for i in range(len(self.treatment_ids)):
            self.weights[i] = np.random.uniform(size=(self.dim_covariates), low=0., high=1.)

    def _init_weights(self):
        self.weights = np.zeros(shape=(len(self.treatment_ids), self.dim_covariates))
        if self.args.treatment_assignment_matrix_distribution == 'uniform':
            self._init_weights_uniform()
        elif self.args.treatment_assignment_matrix_distribution == 'normal':
            self._init_weights_normal()

    def _init_weights_normal(self):
        for i in range(len(self.treatment_ids)):
            self.weights[i] = np.random.multivariate_normal(mean=self.dim_covariates * [0.],
                                                            cov=1. * np.eye(self.dim_covariates),
                                                            size=(1))

    def assign_treatment(self, unit):
        propensity_probabilities = softmax(self.bias * np.matmul(self.weights, self.preprocess_covariates(unit)))
        assigned_treatment = np.random.choice(a=self.treatment_ids, p=propensity_probabilities)
        return assigned_treatment

    def preprocess_covariates(self, covariates):
        if self.policy == 'squared':
            return covariates ** 2
        return covariates

    def get_assignments_for_unit(self, unit, mode: str, num_test_treatments_per_unit: int):
        assignments = None

        if mode == 'most':
            assignments = self.__get_most_likely_assignments_for_unit(unit=unit,
                                                                      num_test_treatments_per_unit=num_test_treatments_per_unit)
        return assignments

    def __get_most_likely_assignments_for_unit(self, unit, num_test_treatments_per_unit: int = 3):
        propensity_probabilities = softmax(np.matmul(self.weights, self.preprocess_covariates(unit)))
        sorted_ids = np.argsort(propensity_probabilities)
        sorted_ids = sorted_ids[-num_test_treatments_per_unit:].tolist()
        propensities_of_selected_treatments = propensity_probabilities[sorted_ids]
        return create_treatment_assignment_dict(all_treatment_ids=self.treatment_ids, sorted_selected_idx=sorted_ids,
                                                propensities_of_selected_treatments=propensities_of_selected_treatments)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
