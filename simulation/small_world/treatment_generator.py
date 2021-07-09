import logging

import networkx as nx
import numpy as np
from tqdm import tqdm

from data.utils import graph_to_edges


def generate_sw_graphs(
    num_graphs: int,
    min_nodes: int,
    max_nodes: int,
    min_neighbours: int,
    max_neighbours: int,
) -> list:
    graphs = []
    logging.info("Starting generating graphs...")
    for i in tqdm(range(num_graphs)):
        num_node = np.random.randint(min_nodes, max_nodes)
        k = np.random.randint(min_neighbours, max_neighbours)
        prob_edge = np.random.uniform(0.1, 1.0)
        graph = nx.connected_watts_strogatz_graph(n=num_node, k=k, p=prob_edge)
        degree_centralities = np.expand_dims(
            list(nx.algorithms.centrality.degree_centrality(graph).values()), axis=1
        )

        # Graph-level features
        node_connectivity = nx.algorithms.node_connectivity(graph)
        aspl = nx.average_shortest_path_length(graph)
        graph_features = [aspl, node_connectivity]
        graphs.append(
            {
                "id": i,
                "graph": graph_to_edges(graph),
                "node_features": degree_centralities,
                "nx_graph": graph,
                "graph_features": graph_features,
            }
        )
    logging.info("Generating graphs done!")
    return graphs
