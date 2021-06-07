import logging
import os
import pickle
import random

import gdown
from rdkit import Chem
from rdkit.Chem import QED
from sklearn.decomposition import KernelPCA
from tqdm import tqdm

from data.utils import normalize_data


def read_csv():
    properties = {}
    with open("./data/tcga/gdb9.sdf.csv", 'r') as f:
        for line in f.readlines()[1:]:
            idx = int(line.split(",")[0].split("_")[1])
            props = line.split(",")
            prop = props[4:]
            properties[idx] = [float(i) for i in prop]
    return properties


def fetch_all_raw_data(num_graphs: int = 500):
    logging.info('reading QM9 data...')
    raw_data = []
    fetch_qm9_data()
    all_files = list(Chem.SDMolSupplier("./data/tcga/gdb9.sdf", False, False, False))
    all_indices = list(range(len(all_files)))
    random.shuffle(all_indices)
    properties = read_csv()
    if num_graphs != -1:
        all_indices = all_indices[:num_graphs]
    for file_idx in tqdm(all_indices):
        mol = all_files[file_idx]
        smiles = Chem.MolToSmiles(mol, kekuleSmiles=True)
        try:
            raw_data.append({'smiles': smiles,
                             'prop': properties[file_idx + 1],
                             'qed': QED.qed(Chem.MolFromSmiles(smiles))
                             })
        except:
            logging.info("bad input %s" % smiles)
    return raw_data


def fetch_qm9_data():
    download_path = './data/tcga/gdb9.tar.gz'
    extract_destination = './data/tcga/'
    if not os.path.exists(download_path):
        logging.info('downloading data to %s ...' % download_path)
        source = 'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/gdb9.tar.gz'
        os.system('wget -O %s %s' % (download_path, source))
        os.system(f'tar xvzf {download_path} -C {extract_destination}')
        logging.info('finished downloading')


def compute_and_save_TCGA_pca(pca_path, n_components):
    tcga_data = get_TCGA_unit_features()
    tcga_pca = kernel_PCA(tcga_data, n_components=n_components)
    pickle.dump(tcga_pca, open(pca_path, 'wb'))
    return tcga_pca


def get_TCGA_pca_data(n_components: int = 10):
    pca_path = './data/tcga/tcga_pca.p'
    if not os.path.exists(pca_path):
        logging.info('TCGA PCA data not found, will generate PCA data now...')
        return compute_and_save_TCGA_pca(pca_path=pca_path, n_components=n_components)
    tcga_pca_data = pickle.load(open(pca_path, 'rb'))
    if tcga_pca_data.shape[1] != n_components:
        logging.info('TCGA PCA data not compatible with number of components, will generate PCA data now...')
        return compute_and_save_TCGA_pca(pca_path=pca_path, n_components=n_components)
    return tcga_pca_data


def kernel_PCA(data, n_components=12, kernel='linear'):
    transformer = KernelPCA(n_components, kernel=kernel)
    data_transformed = transformer.fit_transform(data)
    return data_transformed


def fetch_TCGA_data(path='./data/tcga/tcga.p'):
    if not os.path.exists(path):
        logging.info('TCGA dataset not found, downloading data to %s ...' % path)
        url = 'https://drive.google.com/uc?id=1P-smWytRNuQFjqR403IkJb17CXU6JOM7'
        gdown.download(url, path, quiet=False)
        logging.info('finished downloading')
    return pickle.load(open(path, 'rb'))


def get_TCGA_unit_features(path='./data/tcga/tcga.p'):
    tcga_data = fetch_TCGA_data(path)
    normalized_unit_features = normalize_data(tcga_data['rnaseq'])
    return normalized_unit_features
