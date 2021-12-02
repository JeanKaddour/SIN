from typing import List, Tuple

import numpy as np
from rdkit import Chem

from data.utils import one_of_k_encoding, one_of_k_encoding_unk

# bond mapping
BOND_DICT = {"SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "AROMATIC": 4}


def atom_features(atom) -> np.ndarray:
    return np.array(
        one_of_k_encoding_unk(
            atom.GetSymbol(),
            [
                "C",
                "N",
                "O",
                "S",
                "F",
                "Si",
                "P",
                "Cl",
                "Br",
                "Mg",
                "Na",
                "Ca",
                "Fe",
                "As",
                "Al",
                "I",
                "B",
                "V",
                "K",
                "Tl",
                "Yb",
                "Sb",
                "Sn",
                "Ag",
                "Pd",
                "Co",
                "Se",
                "Ti",
                "Zn",
                "H",
                "Li",
                "Ge",
                "Cu",
                "Au",
                "Ni",
                "Cd",
                "In",
                "Mn",
                "Zr",
                "Cr",
                "Pt",
                "Hg",
                "Pb",
                "Unknown",
            ],
        )
        + one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        + one_of_k_encoding_unk(
            atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        )
        + one_of_k_encoding_unk(
            atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        )
        + [atom.GetIsAromatic()]
    )


def smiles_to_graph(smile: str) -> Tuple[int, List, List, List]:
    mol = Chem.MolFromSmiles(smile)
    edges = []
    edge_types = []
    nodes = []
    for bond in mol.GetBonds():
        edges.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        edge_types.append(BOND_DICT[str(bond.GetBondType())])

    for atom in mol.GetAtoms():
        features = atom_features(atom)
        nodes.append(features / sum(features))

    return mol.GetNumAtoms(), nodes, edges, edge_types
