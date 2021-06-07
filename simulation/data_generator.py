import logging
from abc import ABC
from typing import List

import numpy as np

from data.dataset import Dataset, TestUnit, TestUnits
from simulation.outcome_generators import OutcomeGenerator
from simulation.treatment_assignment import TreatmentAssignmentPolicy


def get_treatment_ids(treatment_assignments):
    return [unit_treatments['treatment_ids'] for unit_treatments in treatment_assignments]


def get_treatment_propensities(treatment_assignments):
    return [unit_treatments['propensities'] for unit_treatments in treatment_assignments]


class AbstractDataGenerator(ABC):
    def __init__(self, id_to_graph_dict, treatment_assignment_policy: TreatmentAssignmentPolicy,
                 outcome_generator: OutcomeGenerator,
                 in_sample_dataset: Dataset, out_sample_dataset: Dataset, args):
        self.id_to_graph_dict = id_to_graph_dict
        self.treatment_assignment_policy = treatment_assignment_policy
        self.outcome_generator = outcome_generator
        self.in_sample_dataset = in_sample_dataset
        self.out_sample_dataset = out_sample_dataset
        self.args = args

    def get_train_assignments(self, units) -> list:
        return [self.treatment_assignment_policy.assign_treatment(unit) for unit in
                units]

    def get_test_assignments(self, units, mode: str,
                             num_test_treatments_per_unit: int) -> list:
        return [self.treatment_assignment_policy.get_assignments_for_unit(unit=unit,
                                                                          num_test_treatments_per_unit=num_test_treatments_per_unit,
                                                                          mode=mode)
                for unit in units]


class DataGenerator(AbstractDataGenerator):

    def __init__(self, id_to_graph_dict, treatment_assignment_policy: TreatmentAssignmentPolicy,
                 outcome_generator: OutcomeGenerator,
                 in_sample_dataset: Dataset, out_sample_dataset: Dataset, args):
        super().__init__(id_to_graph_dict, treatment_assignment_policy, outcome_generator, in_sample_dataset,
                         out_sample_dataset, args)

    def generate_train_data(self) -> None:
        treatment_ids = self.get_train_assignments(units=self.in_sample_dataset.get_units())
        outcomes = self.outcome_generator.generate_outcomes_for_units(units=self.in_sample_dataset.get_units(),
                                                                      treatment_ids=treatment_ids)
        self.in_sample_dataset.add_assigned_treatments(treatment_ids=treatment_ids)
        self.in_sample_dataset.add_outcomes(outcomes=outcomes)

    def get_unseen_treatments(self, in_sample_treatment_assignments, out_sample_treatment_assignments) -> list:
        in_sample_ids = get_treatment_ids(in_sample_treatment_assignments)
        out_sample_ids = get_treatment_ids(out_sample_treatment_assignments)
        all_test_ids = np.concatenate((in_sample_ids, out_sample_ids)).flatten()
        set_test_ids = set(np.unique(all_test_ids))
        set_train_ids = set(self.in_sample_dataset.get_unique_treatment_ids())
        set_unseen_test_ids = set_test_ids - set_train_ids
        return list(set_unseen_test_ids)

    def generate_test_units(self, test_units, test_assignments) -> List[TestUnit]:
        test_data = []
        test_assignments_ids = get_treatment_ids(test_assignments)
        treatment_propensities = get_treatment_propensities(test_assignments)
        for i in range(len(test_units)):
            true_outcomes = self.outcome_generator.generate_outcomes_for_unit(unit=test_units[i],
                                                                              treatment_ids=
                                                                              test_assignments_ids[i])

            test_unit = TestUnit(covariates=test_units[i], treatment_ids=test_assignments_ids[i],
                                 treatment_propensities=treatment_propensities[i],
                                 true_outcomes=true_outcomes)
            test_data.append(test_unit)
        return test_data

    def generate_test_data(self) -> TestUnits:
        in_sample_units, out_sample_units = self.in_sample_dataset.get_units(), self.out_sample_dataset.get_units()
        logging.info(f'Num in-sample units: {len(in_sample_units)}')
        logging.info(f'Num out-sample units: {len(out_sample_units)}')

        in_sample_treatment_assignments = self.get_test_assignments(units=in_sample_units,
                                                                    mode='most',
                                                                    num_test_treatments_per_unit=self.args.max_test_assignments)
        out_sample_treatment_assignments = self.get_test_assignments(units=out_sample_units,
                                                                     mode='most',
                                                                     num_test_treatments_per_unit=self.args.max_test_assignments)
        in_sample_test_units = self.generate_test_units(test_units=in_sample_units,
                                                        test_assignments=in_sample_treatment_assignments)
        out_sample_test_units = self.generate_test_units(test_units=out_sample_units,
                                                         test_assignments=out_sample_treatment_assignments)
        test_units_dict = {'in_sample': in_sample_test_units, 'out_sample': out_sample_test_units}
        unseen_treatment_ids = self.get_unseen_treatments(
            in_sample_treatment_assignments=in_sample_treatment_assignments,
            out_sample_treatment_assignments=out_sample_treatment_assignments)
        return TestUnits(test_units_dict=test_units_dict,
                         id_to_graph_dict=self.id_to_graph_dict, unseen_treatment_ids=unseen_treatment_ids)
