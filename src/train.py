import os
import pandas as pd
import numpy as np
import pickle
from scipy.spatial import distance_matrix
import multiprocessing
from itertools import repeat
import networkx as nx
import torch
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit import RDLogger
from torch_geometric.data import Batch, Data
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
import warnings
from model import model
from config.config_dict import Config
from log.train_logger import TrainLogger
from utils import *

RDLogger.DisableLog('rdApp.*')
np.set_printoptions(threshold=np.inf)
warnings.filterwarnings('ignore')

def one_hot_encoding(k, possible_values):
    if k not in possible_values:
        raise ValueError(f"Invalid value: {k} not in {possible_values}")
    return [k == e for e in possible_values]

def one_hot_encoding_with_default(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def extract_atom_features(mol, graph, atom_symbols=['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I'], include_hydrogens=True):
    for atom in mol.GetAtoms():
        features = one_hot_encoding_with_default(atom.GetSymbol(), atom_symbols + ['Unknown']) + \
                  one_hot_encoding_with_default(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]) + \
                  one_hot_encoding_with_default(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
                  one_hot_encoding_with_default(atom.GetHybridization(), [
                      Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                      Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                      Chem.rdchem.HybridizationType.SP3D2]) + [atom.GetIsAromatic()]
        if include_hydrogens:
            features += one_hot_encoding_with_default(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
        atom_features = np.array(features).astype(np.float32)
        graph.add_node(atom.GetIdx(), feats=torch.from_numpy(atom_features))

def generate_edge_index(mol, graph):
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        graph.add_edge(i, j)

def convert_mol_to_graph(mol):
    graph = nx.Graph()
    extract_atom_features(mol, graph)
    generate_edge_index(mol, graph)
    graph = graph.to_directed()
    node_features = torch.stack([feats['feats'] for _, feats in graph.nodes(data=True)])
    edge_index = torch.stack([torch.LongTensor((u, v)) for u, v in graph.edges(data=False)]).T
    return node_features, edge_index

def create_interaction_graph(ligand, pocket, distance_threshold=5.):
    ligand_atoms = ligand.GetNumAtoms()
    pocket_atoms = pocket.GetNumAtoms()
    interaction_graph = nx.Graph()
    ligand_positions = ligand.GetConformers()[0].GetPositions()
    pocket_positions = pocket.GetConformers()[0].GetPositions()
    distance_matrix = distance_matrix(ligand_positions, pocket_positions)
    close_nodes = np.where(distance_matrix < distance_threshold)
    for i, j in zip(close_nodes[0], close_nodes[1]):
        interaction_graph.add_edge(i, j + ligand_atoms)
    interaction_graph = interaction_graph.to_directed()
    interaction_edge_index = torch.stack([torch.LongTensor((u, v)) for u, v in interaction_graph.edges(data=False)]).T
    return interaction_edge_index

def process_molecular_complex(complex_path, label, output_path, distance_threshold=5.):
    with open(complex_path, 'rb') as f:
        ligand, pocket = pickle.load(f)
    ligand_atoms = ligand.GetNumAtoms()
    pocket_atoms = pocket.GetNumAtoms()
    ligand_positions = torch.FloatTensor(ligand.GetConformers()[0].GetPositions())
    pocket_positions = torch.FloatTensor(pocket.GetConformers()[0].GetPositions())
    ligand_features, ligand_edge_index = convert_mol_to_graph(ligand)
    pocket_features, pocket_edge_index = convert_mol_to_graph(pocket)
    combined_features = torch.cat([ligand_features, pocket_features], dim=0)
    intra_edge_index = torch.cat([ligand_edge_index, pocket_edge_index + ligand_atoms], dim=-1)
    interaction_edge_index = create_interaction_graph(ligand, pocket, distance_threshold=distance_threshold)
    target = torch.FloatTensor([label])
    positions = torch.cat([ligand_positions, pocket_positions], dim=0)
    split = torch.cat([torch.zeros((ligand_atoms, )), torch.ones((pocket_atoms,))], dim=0)
    graph_data = Data(x=combined_features, edge_index_intra=intra_edge_index, edge_index_inter=interaction_edge_index, y=target, pos=positions, split=split)
    torch.save(graph_data, output_path)

class MolecularGraphDataLoader(DataLoader):
    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, collate_fn=dataset.collate_fn, **kwargs)

class MolecularGraphDataset(Dataset):
    def __init__(self, data_directory, data_frame, distance_threshold=5, graph_type='Graph_HIGAN', num_processes=8, create=False):
        self.data_directory = data_directory
        self.data_frame = data_frame
        self.distance_threshold = distance_threshold
        self.graph_type = graph_type
        self.create = create
        self.num_processes = num_processes
        self.graph_paths = None
        self.complex_ids = None
        self._initialize_graphs()

    def _initialize_graphs(self):
        complex_paths = []
        complex_ids = []
        labels = []
        graph_paths = []
        thresholds = repeat(self.distance_threshold, len(self.data_frame))
        for i, row in self.data_frame.iterrows():
            complex_id, label = row['pdbid'], float(row['-logKd/Ki'])
            complex_directory = os.path.join(self.data_directory, complex_id)
            graph_path = os.path.join(complex_directory, f"{self.graph_type}-{complex_id}_{self.distance_threshold}A.pyg")
            complex_path = os.path.join(complex_directory, f"{complex_id}_{self.distance_threshold}A.rdkit")
            complex_paths.append(complex_path)
            complex_ids.append(complex_id)
            labels.append(label)
            graph_paths.append(graph_path)
        if self.create:
            pool = multiprocessing.Pool(self.num_processes)
            pool.starmap(process_molecular_complex, zip(complex_paths, labels, graph_paths, thresholds))
            pool.close()
            pool.join()
        self.graph_paths = graph_paths
        self.complex_ids = complex_ids

    def __getitem__(self, idx):
        return torch.load(self.graph_paths[idx])

    def collate_fn(self, batch):
        return Batch.from_data_list(batch)

    def __len__(self):
        return len(self.data_frame)

class PerformanceMeter(object):
    def __init__(self, mode):
        self.mode = mode
        self.reset()

    def reset(self):
        self.best = float('inf') if self.mode == 'min' else -float('inf')
        self.count = 0

    def update(self, best_value):
        self.best = best_value
        self.count = 0

    def get_best(self):
        return self.best

    def increment_counter(self):
        self.count += 1
        return self.count

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.val = value
        self.sum += value * n
        self.count += n

    def get_average(self):
        return self.sum / (self.count + 1e-12)

def validate_model(model, dataloader, device):
    model.eval()
    predictions, ground_truths = [], []
    for data in dataloader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data)
            label = data.y
            predictions.append(pred.cpu().numpy())
            ground_truths.append(label.cpu().numpy())
    pred = np.concatenate(predictions, axis=0)
    label = np.concatenate(ground_truths, axis=0)
    correlation = np.corrcoef(pred, label)[0, 1]
    rmse = np.sqrt(mean_squared_error(label, pred))
    model.train()
    return rmse, correlation

if __name__ == '__main__':
    config_file = 'TrainConfig_HIGAN'
    config = Config(config_file)
    args = config.get_config()

    for epoch in range(args['repeat']):
        train_dir = os.path.join(args['data_root'], 'train')
        valid_dir = os.path.join(args['data_root'], 'valid')
        test2016_dir = os.path.join(args['data_root'], 'test2016')

        train_df = pd.read_csv(os.path.join(args['data_root'], 'train.csv'))
        valid_df = pd.read_csv(os.path.join(args['data_root'], 'valid.csv'))
        test2016_df = pd.read_csv(os.path.join(args['data_root'], 'test2016.csv'))

        train_dataset = MolecularGraphDataset(train_dir, train_df, graph_type=args['graph_type'], create=False)
        valid_dataset = MolecularGraphDataset(valid_dir, valid_df, graph_type=args['graph_type'], create=False)
        test2016_dataset = MolecularGraphDataset(test2016_dir, test2016_df, graph_type=args['graph_type'], create=False)

        train_loader = MolecularGraphDataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=4)
        valid_loader = MolecularGraphDataLoader(valid_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=4)

        logger = TrainLogger(args, config_file, create=True)
        logger.info("Training started...")

        device = torch.device('cuda:0')
        model = HIGAN(35, 256).to(device)
        optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-6)
        criterion = nn.MSELoss()

        best_meter = PerformanceMeter("min")
        for epoch in range(args['epochs']):
            model.train()
            for data in train_loader:
                data = data.to(device)
                predictions = model(data)
                labels = data.y
                loss = criterion(predictions, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            validation_rmse, validation_corr = validate_model(model, valid_loader, device)
            logger.info(f"Epoch {epoch}, Validation RMSE: {validation_rmse}, Validation Pearson Correlation: {validation_corr}")

            if validation_rmse < best_meter.get_best():
                best_meter.update(validation_rmse)

        logger.info("Final Testing Results:")
        validation_rmse, validation_corr = validate_model(model, valid_loader, device)
        logger.info(f"Final Validation RMSE: {validation_rmse}, Final Pearson Correlation: {validation_corr}")
