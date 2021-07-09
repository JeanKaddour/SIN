import argparse
from datetime import datetime

import torch

import wandb
from configs.run_model import sw, tcga
from configs.utils import str2bool
from experiments.logger import create_logger
from experiments.train import train_and_test
from experiments.utils import init_seeds, save_run_results

TIME_STR = "{:%Y_%m_%d_%H_%M_%S_%f}".format(datetime.now())
DATE_STR = "{:%Y_%m_%d}".format(datetime.now())


def parse_default_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GraphInterventionNetworks")
    parser.add_argument("--name", type=str, default=TIME_STR)
    parser.add_argument("--task", type=str, default="sw", choices=["sw", "tcga"])
    parser.add_argument(
        "--model",
        type=str,
        default="gin",
        choices=["gin", "gnn", "graphite", "cat", "zero"],
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument(
        "--log_interval",
        type=int,
        default=50,
        help="How many batches to wait before logging training status",
    )
    parser.add_argument(
        "--ablation", type=str2bool, default=False, help="Changes wandb project name."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./generated_data/",
        help="Path to save/load generated data",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="./results/",
        help="Path to save experimental results",
    )
    args, _ = parser.parse_known_args()

    if args.task == "sw":
        sw.add_params(parser)
    elif args.task == "tcga":
        tcga.add_params(parser)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_default_args()
    project_name = (
        f"GIN_{DATE_STR}-{args.task}-ABL"
        if args.ablation
        else f"GIN_{DATE_STR}-{args.task}"
    )
    wandb.init(project=project_name, name=args.model + "-" + str(args.seed))

    wandb.config.update(args)
    init_seeds(seed=args.seed)

    logger = create_logger("log/%s.log" % args.name)
    logger.info(args)

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    test_units_with_predictions, test_errors = train_and_test(args=args, device=device)
    save_run_results(
        test_units_with_predictions=test_units_with_predictions,
        test_errors=test_errors,
        time_str=TIME_STR,
        args=args,
    )
