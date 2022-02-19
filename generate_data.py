import argparse
from datetime import datetime

from configs.generate_data import sw, tcga
from experiments.io import save_dataset
from experiments.logger import create_logger
from experiments.utils import init_seeds
from simulation.utils import (create_dataset, get_data_generator,
                              get_outcome_generator,
                              get_treatment_assignment_policy)


def parse_default_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GraphInterventionNetworks")
    parser.add_argument(
        "--name", type=str, default="{:%Y_%m_%d_%H_%M_%S_%f}".format(datetime.now())
    )
    parser.add_argument("--task", type=str, default="tcga", choices=["sw", "tcga"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument(
        "--data_path",
        type=str,
        default="./generated_data/",
        help="Path to save/load generated data",
    )

    args, _ = parser.parse_known_args()
    if args.task == "sw":
        sw.add_params(parser)
    elif args.task == "tcga":
        tcga.add_params(parser)
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_default_args()
    logger = create_logger("log/%s.log" % args.name)
    logger.info(args)
    init_seeds(args.seed)
    in_sample_dataset, out_sample_dataset, id_to_graph_dict = create_dataset(args)
    all_treatment_ids = list(id_to_graph_dict.keys())
    treatment_assignment_policy = get_treatment_assignment_policy(
        treatment_ids=all_treatment_ids, args=args
    )
    outcome_generator = get_outcome_generator(
        id_to_graph_dict=id_to_graph_dict, args=args
    )
    data_generator = get_data_generator(
        task=args.task,
        id_to_graph_dict=id_to_graph_dict,
        treatment_assignment_policy=treatment_assignment_policy,
        outcome_generator=outcome_generator,
        in_sample_dataset=in_sample_dataset,
        out_sample_dataset=out_sample_dataset,
        args=args,
    )
    logger.info("Generate outcomes...")
    data_generator.generate_train_data()
    test_units = data_generator.generate_test_data()
    save_dataset(in_sample_dataset=in_sample_dataset, test_units=test_units, args=args)


if __name__ == "__main__":
    main()
