import logging
from argparse import Namespace
from typing import Callable, Tuple, Union

from data.dataset import Dataset
from simulation.data_generator import DataGenerator
from simulation.outcome_generators import OutcomeGenerator
from simulation.small_world.outcome_simulator import SmallWorldSimulator
from simulation.tcga.data_generator import TCGADataGenerator
from simulation.tcga.outcome_simulator import TCGASimulator
from simulation.treatment_assignment import RandomTAP, TreatmentAssignmentPolicy
from simulation.treatment_generators import (
    generate_id_to_graph_dict_sw,
    generate_id_to_graph_dict_tcga,
)
from simulation.unit_generators import (
    generate_TCGA_unit_features,
    generate_uniform_unit_features,
)


def get_treatment_assignment_policy(treatment_ids: list, args: Namespace) -> RandomTAP:
    return RandomTAP(treatment_ids=treatment_ids, args=args)


def get_outcome_generator(
    id_to_graph_dict: dict, args: Namespace
) -> Union[None, OutcomeGenerator]:
    outcome_generator = None

    if args.task == "sw":
        outcome_generator = SmallWorldSimulator(
            id_to_graph_dict=id_to_graph_dict,
            noise_mean=args.outcome_noise_mean,
            noise_std=args.outcome_noise_std,
            dim_covariates=args.dim_covariates,
        )
    elif args.task == "tcga":
        outcome_generator = TCGASimulator(
            id_to_graph_dict=id_to_graph_dict,
            noise_mean=args.outcome_noise_mean,
            noise_std=args.outcome_noise_std,
            dim_covariates=args.dim_covariates,
        )

    return outcome_generator


def get_data_generator(
    task: str,
    id_to_graph_dict: dict,
    treatment_assignment_policy: TreatmentAssignmentPolicy,
    outcome_generator: OutcomeGenerator,
    in_sample_dataset: Dataset,
    out_sample_dataset: Dataset,
    args: Namespace,
) -> DataGenerator:
    data_generator = None
    if task == "sw":
        data_generator = DataGenerator(
            id_to_graph_dict=id_to_graph_dict,
            treatment_assignment_policy=treatment_assignment_policy,
            outcome_generator=outcome_generator,
            in_sample_dataset=in_sample_dataset,
            out_sample_dataset=out_sample_dataset,
            args=args,
        )
    elif task == "tcga":
        data_generator = TCGADataGenerator(
            id_to_graph_dict=id_to_graph_dict,
            treatment_assignment_policy=treatment_assignment_policy,
            outcome_generator=outcome_generator,
            in_sample_dataset=in_sample_dataset,
            out_sample_dataset=out_sample_dataset,
            args=args,
        )

    return data_generator


def get_unit_generator(args: Namespace) -> Callable:
    if args.task == "tcga":
        return generate_TCGA_unit_features
    unit_generator = None
    if args.unit_distribution == "uniform":
        unit_generator = generate_uniform_unit_features
    return unit_generator


def get_treatment_generator(args: Namespace) -> Callable:
    treatment_generator = None
    if args.task == "sw":
        treatment_generator = generate_id_to_graph_dict_sw
    elif args.task == "tcga":
        treatment_generator = generate_id_to_graph_dict_tcga
    return treatment_generator


def create_dataset_dicts(
    unit_generator, treatment_generator, args: Namespace
) -> Tuple[dict, dict, dict]:
    in_sample_dataset_dict, out_sample_dataset_dict = {}, {}
    logging.info("Generate units...")
    in_sample_dataset_dict["units"], out_sample_dataset_dict["units"] = unit_generator(
        args=args
    )
    logging.info("Generate treatments...")
    id_to_graph_dict = treatment_generator(args=args)
    in_sample_dataset_dict["id_to_graph_dict"] = id_to_graph_dict
    return in_sample_dataset_dict, out_sample_dataset_dict, id_to_graph_dict


def create_dataset(args: Namespace) -> Tuple[Dataset, Dataset, dict]:
    unit_generator = get_unit_generator(args=args)
    treatment_generator = get_treatment_generator(args=args)
    (
        in_sample_dataset_dict,
        out_sample_dataset_dict,
        id_to_graph_dict,
    ) = create_dataset_dicts(
        unit_generator=unit_generator,
        treatment_generator=treatment_generator,
        args=args,
    )
    in_sample_dataset, out_sample_dataset = Dataset(
        data_dict=in_sample_dataset_dict
    ), Dataset(data_dict=out_sample_dataset_dict)
    return in_sample_dataset, out_sample_dataset, id_to_graph_dict
