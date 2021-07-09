import logging
import pickle
from argparse import Namespace
from pathlib import Path

from data.dataset import Dataset, TestUnits


def load_train_dataset(args: Namespace) -> Dataset:
    file_path = args.data_path + f"{args.task}/seed-{args.seed}/bias-{args.bias}/"
    dataset = pickle.load(open(file_path + "in_sample.p", "rb"))
    return dataset


def load_test_dataset(args: Namespace) -> TestUnits:
    file_path = args.data_path + f"{args.task}/seed-{args.seed}/bias-{args.bias}/"
    test_units = pickle.load(open(file_path + "test.p", "rb"))
    return test_units


def pickle_dump(file_name: str, content: object) -> None:
    with open(file_name, "wb") as out_file:
        pickle.dump(content, out_file, pickle.HIGHEST_PROTOCOL)


def save_dataset(
    in_sample_dataset: Dataset, test_units: TestUnits, args: Namespace
) -> None:
    file_path = args.data_path + f"{args.task}/seed-{args.seed}/bias-{args.bias}/"
    Path(file_path).mkdir(parents=True, exist_ok=True)
    pickle_dump(file_name=file_path + "in_sample.p", content=in_sample_dataset)
    pickle_dump(file_name=file_path + "test.p", content=test_units)
    logging.info(f"Saved training and test dataset successfully.")
