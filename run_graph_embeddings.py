import argparse
from datetime import datetime

import torch

import wandb
from configs.run_model import sw, tcga
from data.utils import one_of_k_encoding
from experiments.io import load_train_dataset, load_test_dataset, pickle_dump
from experiments.logger import create_logger
from experiments.train import train
from experiments.utils import compute_graph_embeddings, get_ids_with_closest_distance, get_train_and_val_dataset
from experiments.utils import get_model
from experiments.utils import init_seeds

time_str = '{:%Y_%m_%d_%H_%M_%S_%f}'.format(datetime.now())
date_str = '{:%Y_%m_%d}'.format(datetime.now())


def parse_default_args():
    parser = argparse.ArgumentParser(description='GraphInterventionNetworks')
    parser.add_argument('--name', type=str, default=time_str)
    parser.add_argument('--task', type=str, default='sw', choices=['sw', 'tcga'])
    parser.add_argument('--model', type=str, default='gnn',
                        choices=['gnn'])
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--log_interval', type=int, default=10,
                        help='How many batches to wait before logging training status')

    parser.add_argument('--data_path', type=str, default='./generated_data/', help='Path to save/load generated data')
    args, _ = parser.parse_known_args()
    if args.task == 'sw':
        sw.add_params(parser)
    elif args.task == 'tcga':
        tcga.add_params(parser)
    args = parser.parse_args()
    return args


def update_one_hot_encodings(args, id_to_graph_dict, unseen_treatment_ids, closest_graph_ids, test_units):
    all_treatment_ids = [treatment_id for treatment_id in id_to_graph_dict.keys()]
    if args.task == 'tcga':
        all_treatment_ids = list(range(args.num_treatments))
    for unseen_id, seen_id in zip(unseen_treatment_ids, closest_graph_ids):
        id_to_graph_dict[unseen_id]['one_hot_encoding'] = one_of_k_encoding(x=seen_id, allowable_set=all_treatment_ids)
    test_units.set_id_to_graph_dict(id_to_graph_dict=id_to_graph_dict)
    file_path = args.data_path + f'{args.task}/seed-{args.seed}/bias-{args.bias}/'
    pickle_dump(file_name=file_path + "test.p", content=test_units)


if __name__ == "__main__":
    args = parse_default_args()
    wandb.init(project=f"GIN_EMB_{date_str}-{args.task}", name=args.model + "-" + str(args.seed))
    wandb.config.update(args)
    init_seeds(seed=args.seed)
    logger = create_logger('log/%s.log' % args.name)
    logger.info(args)

    test_units = load_test_dataset(args=args)
    unseen_treatment_ids = test_units.get_unseen_treatment_ids()
    if len(unseen_treatment_ids) > 0:
        logger.info(f'Unseen treatment ids: {unseen_treatment_ids}')
        device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
        model = get_model(args=args, device=device)
        train_dataset_pt, val_dataset_pt = get_train_and_val_dataset(args=args)
        train(model=model, train_dataset_pt=train_dataset_pt, val_dataset_pt=val_dataset_pt, device=device, args=args)

        in_sample_dataset = load_train_dataset(args=args)
        seen_treatment_ids = in_sample_dataset.get_unique_treatment_ids()
        id_to_graph_dict = in_sample_dataset.get_id_to_graph_dict()
        graph_embeddings_seen_treatments = compute_graph_embeddings(model=model, device=device,
                                                                    treatment_ids=seen_treatment_ids,
                                                                    id_to_graph_dict=id_to_graph_dict)
        graph_embeddings_unseen_treatments = compute_graph_embeddings(model=model, device=device,
                                                                      treatment_ids=unseen_treatment_ids,
                                                                      id_to_graph_dict=id_to_graph_dict)
        # Compute distances between embeddings and find nearest ones
        closest_graph_ids = get_ids_with_closest_distance(target_embeddings=graph_embeddings_unseen_treatments,
                                                          source_embeddings=graph_embeddings_seen_treatments,
                                                          source_ids=seen_treatment_ids)
        logger.info(f'Closest treatment ids to unseen treatment ids: {closest_graph_ids}')
        # Update one-hot encodings of id_to_graph_dict
        update_one_hot_encodings(args=args, id_to_graph_dict=id_to_graph_dict,
                                 unseen_treatment_ids=unseen_treatment_ids,
                                 closest_graph_ids=closest_graph_ids, test_units=test_units)
        logger.info('Successfully updated one-hot encodings of unseen treatments')
    else:
        logger.info('There are no unseen treatments.')
