import argparse
import logging

from tqdm import tqdm

from data.tcga.qm9_tcga_simulation import fetch_all_raw_data
from data.tcga.smiles_processing import smiles_to_graph
from data.utils import one_of_k_encoding
from simulation.small_world.treatment_generator import generate_sw_graphs


def generate_id_to_graph_dict_sw(args: argparse.Namespace) -> dict:
    treatments_list = generate_sw_graphs(
        num_graphs=args.num_graphs,
        min_nodes=args.min_num_nodes,
        max_nodes=args.max_num_nodes,
        min_neighbours=args.min_neighbours,
        max_neighbours=args.max_neighbours,
    )
    all_treatment_ids = list(range(len(treatments_list)))
    id_to_graph_dict = {}
    for treatment in treatments_list:
        id = treatment["id"]
        id_to_graph_dict[id] = {
            "c_size": len(treatment["node_features"]),
            "node_features": treatment["node_features"],
            "edges": treatment["graph"],
            "graph_features": treatment["graph_features"],
            "one_hot_encoding": one_of_k_encoding(
                x=id, allowable_set=all_treatment_ids
            ),
        }
    return id_to_graph_dict


def generate_id_to_graph_dict_tcga(args: argparse.Namespace) -> dict:
    raw_data = fetch_all_raw_data(num_graphs=args.num_graphs)
    id_to_graph_dict = {}
    all_ids = list(range(args.num_graphs))
    logging.info("Convert smiles data to graphs...")

    for i, raw_molecule in enumerate(tqdm(raw_data)):
        c_size, features, edge_index, edge_types = smiles_to_graph(
            raw_molecule["smiles"]
        )
        if any(i <= 1 for i in [c_size, len(features), len(edge_types), len(edge_index)]):
            continue
        id_to_graph_dict[i] = {
            "c_size": c_size,
            "node_features": features,
            "edge_types": edge_types,
            "edges": edge_index,
            "prop": raw_molecule["prop"],
            "qed": raw_molecule["qed"],
            "one_hot_encoding": one_of_k_encoding(x=i, allowable_set=all_ids),
        }

    return id_to_graph_dict
