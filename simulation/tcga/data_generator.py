import numpy as np

from data.dataset import Dataset, TestUnit, TestUnits
from simulation.data_generator import (DataGenerator, get_treatment_ids,
                                       get_treatment_propensities)
from simulation.outcome_generators import OutcomeGenerator
from simulation.treatment_assignment import TreatmentAssignmentPolicy


class TCGADataGenerator(DataGenerator):
    def __init__(
        self,
        id_to_graph_dict,
        treatment_assignment_policy: TreatmentAssignmentPolicy,
        outcome_generator: OutcomeGenerator,
        in_sample_dataset: Dataset,
        out_sample_dataset: Dataset,
        args,
    ):
        super().__init__(
            id_to_graph_dict,
            treatment_assignment_policy,
            outcome_generator,
            in_sample_dataset,
            out_sample_dataset,
            args,
        )

    def generate_train_data(self) -> None:
        treatment_ids = self.get_train_assignments(
            units=self.in_sample_dataset.get_units()["features"]
        )
        outcomes = self.outcome_generator.generate_outcomes_for_units(
            pca_features=self.in_sample_dataset.get_units()["pca_features"],
            unit_features=self.in_sample_dataset.get_units()["features"],
            treatment_ids=treatment_ids,
        )
        self.in_sample_dataset.add_assigned_treatments(treatment_ids=treatment_ids)
        self.in_sample_dataset.add_outcomes(outcomes=outcomes)

    def get_unseen_treatments(
        self, in_sample_treatment_assignments, out_sample_treatment_assignments
    ):
        in_sample_ids = get_treatment_ids(in_sample_treatment_assignments)
        out_sample_ids = get_treatment_ids(out_sample_treatment_assignments)
        all_test_ids = np.concatenate((in_sample_ids, out_sample_ids)).flatten()
        set_test_ids = set(np.unique(all_test_ids))
        set_train_ids = set(self.in_sample_dataset.get_unique_treatment_ids())
        set_unseen_test_ids = set_test_ids - set_train_ids
        return list(set_unseen_test_ids)

    def generate_test_units(self, test_units, test_assignments):
        test_data = []
        test_assignments_ids = get_treatment_ids(test_assignments)
        treatment_propensities = get_treatment_propensities(test_assignments)
        for i in range(len(test_units["ids"])):
            true_outcomes = self.outcome_generator.generate_outcomes_for_unit(
                pca_features=test_units["pca_features"][i],
                unit_features=test_units["features"][i],
                treatment_ids=test_assignments_ids[i],
            )

            test_unit = TestUnit(
                covariates=test_units["features"][i],
                treatment_ids=test_assignments_ids[i],
                treatment_propensities=treatment_propensities[i],
                true_outcomes=true_outcomes,
            )
            test_data.append(test_unit)
        return test_data

    def generate_test_data(self):
        in_sample_units, out_sample_units = (
            self.in_sample_dataset.get_units(),
            self.out_sample_dataset.get_units(),
        )
        print(f'Num in-sample units: {len(in_sample_units["ids"])}')
        print(f'Num out-sample units: {len(out_sample_units["ids"])}')

        in_sample_treatment_assignments = self.get_test_assignments(
            units=in_sample_units["features"],
            mode="most",
            num_test_treatments_per_unit=self.args.max_test_assignments,
        )
        out_sample_treatment_assignments = self.get_test_assignments(
            units=out_sample_units["features"],
            mode="most",
            num_test_treatments_per_unit=self.args.max_test_assignments,
        )
        in_sample_test_units = self.generate_test_units(
            test_units=in_sample_units, test_assignments=in_sample_treatment_assignments
        )
        out_sample_test_units = self.generate_test_units(
            test_units=out_sample_units,
            test_assignments=out_sample_treatment_assignments,
        )
        test_units_dict = {
            "in_sample": in_sample_test_units,
            "out_sample": out_sample_test_units,
        }
        unseen_treatment_ids = self.get_unseen_treatments(
            in_sample_treatment_assignments=in_sample_treatment_assignments,
            out_sample_treatment_assignments=out_sample_treatment_assignments,
        )
        return TestUnits(
            test_units_dict=test_units_dict,
            id_to_graph_dict=self.id_to_graph_dict,
            unseen_treatment_ids=unseen_treatment_ids,
        )
