import os, json, ast, glob, ssl
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import torch
import os.path as osp
import numpy as np
import pandas as pd
from torch_geometric.transforms import Compose

from tqdm import tqdm
from rdkit import Chem
from itertools import repeat
from six.moves import urllib
from torch_geometric.data import Data, InMemoryDataset, download_url

from .molx_utils import mol2graph


def process_supplier_chunk(args):
    i, sdf_path, start, end, sdf_paths, node_features, edge_features = args
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=True)

    data_list = []

    for j in tqdm(range(start, end), desc=f'{i + 1}/{len(sdf_paths)}'):
        mol = suppl[j]
        coords = mol.GetConformer().GetPositions()
        z = [atom.GetAtomicNum() for atom in mol.GetAtoms()]

        if  node_features or edge_features:
            graph = mol2graph(mol)
        data_dict = {
            'num_nodes': coords.shape[0],
            'z': z, # atomic number
            'xyz': coords
        }
        if edge_features:
            data_dict['edge_attr'] = graph['edge_feat']
            data_dict['edge_feat'] = graph['edge_feat']
        if node_features:
            data_dict['x'] = graph['node_feat']
            smiles = Chem.MolToSmiles(mol)
            data_dict['smiles'] = smiles

        data_list.append(data_dict)

    return data_list

m3d_target_dict = {
    0: "mu",
    1: "homo",
    2: "lumo",
    3: "gap",
}

from multiprocessing import Pool, cpu_count
class Molecule3D(InMemoryDataset):
    mu = "mu"
    homo = "homo"
    lumo = "lumo"
    gap = "gap"
    muxyz = "muxyz"

    available_properties = [
        mu,
        homo,
        lumo,
        gap,
        muxyz,
    ]

    def __init__(self,
                 root,
                 split='train',
                 split_mode='random',
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 data_folder='M3D',
                 edge_features=False,
                 node_features=False,
                label=None,
                 ):

        assert split in ['train', 'val', 'test', 'all']
        assert split_mode in ['random', 'scaffold']
        self.split_mode = split_mode
        self.root = root
        self.name = data_folder
        self.label = label
        if isinstance(label, int):
            self.label_idx = label
        else:
            label2idx = dict(zip(m3d_target_dict.values(), m3d_target_dict.keys()))
            self.label_idx = label2idx[self.label]
        self.target_df = pd.read_csv(osp.join(self.raw_dir, 'properties.csv'))
        self.node_features = node_features
        self.edge_features = edge_features
        # if not osp.exists(self.raw_paths[0]):
        #     self.download()

        if transform is None:
            transform = self._filter_label
        else:
            transform = Compose([transform, self._filter_label])

        super(Molecule3D, self).__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(
            osp.join(self.processed_dir, '{}_{}.pt'.format(split_mode, split)))

    @property
    def num_node_labels(self):
        if self.data.x is None:
            return 0
        for i in range(self.data.x.size(1)):
            x = self.data.x[:, i:]
            if ((x == 0) | (x == 1)).all() and (x.sum(dim=1) == 1).all():
                return self.data.x.size(1) - i
        return 0

    @property
    def num_node_attributes(self):
        if self.data.x is None:
            return 0
        return self.data.x.size(1) - self.num_node_labels

    @property
    def num_edge_labels(self):
        if self.data.edge_attr is None:
            return 0
        for i in range(self.data.edge_attr.size(1)):
            if self.data.edge_attr[:, i:].sum() == self.data.edge_attr.size(0):
                return self.data.edge_attr.size(1) - i
        return 0

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        name = self.name + '.csv'
        return name

    @property
    def processed_file_names(self):
        return ['random_train.pt', 'random_val.pt', 'random_test.pt',
                'scaffold_train.pt', 'scaffold_val.pt', 'scaffold_test.pt']

    def download(self):
        # print('making raw files:', self.raw_dir)
        # if not osp.exists(self.raw_dir):
        #     os.makedirs(self.raw_dir)
        # url = self.url
        # path = download_url(url, self.raw_dir)
        pass

    def pre_process(self):
        sdf_paths = [osp.join(self.raw_dir, 'combined_mols_0_to_1000000.sdf'),
                     osp.join(self.raw_dir, 'combined_mols_1000000_to_2000000.sdf'),
                     osp.join(self.raw_dir, 'combined_mols_2000000_to_3000000.sdf'),
                     osp.join(self.raw_dir, 'combined_mols_3000000_to_3899647.sdf')]

        mol_counts = [len(Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=True)) for sdf_path in sdf_paths]

        chunk_size = 1000000 // 64  # You can adjust this value depending on the granularity you need
        args = [(i, sdf_path, start, min(start + chunk_size, mol_count), sdf_paths, self.node_features, self.edge_features)
                for i, (sdf_path, mol_count) in enumerate(zip(sdf_paths, mol_counts))
                for start in range(0, mol_count, chunk_size)]

        data_dict_list = []
        with ProcessPoolExecutor() as executor:
            for result in executor.map(process_supplier_chunk, args, chunksize=1):
                data_dict_list.extend(result)

        print("Processing minified v4.")
        # Convert dictionaries to Data instances and numpy arrays to torch tensors
        data_list = []
        for abs_idx, data_dict in enumerate(tqdm(data_dict_list)):
            data = Data()
            # print(data_dict['num_nodes'])
            data.nodes = torch.tensor(data_dict['num_nodes'], dtype=torch.int16)
            if self.edge_features:
                data.edge_index = torch.from_numpy(data_dict['edge_index']).to(torch.int32)
                data.edge_attr = torch.from_numpy(data_dict['edge_attr']).to(torch.int32)
            if self.node_features:
                data.smiles = data_dict['smiles']
                data.x = torch.from_numpy(data_dict['x']).to(torch.int32)
            data.z = torch.tensor(data_dict['z'], dtype=torch.int16)
            data.pos = torch.tensor(data_dict['xyz'], dtype=torch.float32)

            # Properties
            targets = self.target_df.iloc[abs_idx, 1:].values
            dipole_xyz = targets[:3]
            dipole_moment = np.linalg.norm(dipole_xyz)
            labels = torch.tensor([dipole_moment, *targets[3:]], dtype=torch.float32).unsqueeze(0)

            data.labels = labels
            data.dipole_xyz = torch.tensor(dipole_xyz, dtype=torch.float32).unsqueeze(0)
            data_list.append(data)

        return data_list

    def _filter_label(self, batch):
        batch.y = batch.labels[:, self.label_idx].unsqueeze(1)
        return batch

    def process(self):
        full_list = self.pre_process()

        # process all
        data_list = full_list
        if self.pre_filter is not None:
            data_list = [data for data in full_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        torch.save(self.collate(data_list), self.processed_paths[-1])


        print('making processed files:', self.processed_dir)
        if not osp.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

        for m, split_mode in enumerate(['random', 'scaffold']):
            ind_path = osp.join(self.raw_dir, '{}_split_inds.json').format(split_mode)
            with open(ind_path, 'r') as f:
                inds = json.load(f)

            for s, split in enumerate(['train', 'valid', 'test']):
                data_list = [full_list[idx] for idx in inds[split]]
                if self.pre_filter is not None:
                    data_list = [data for data in data_list if self.pre_filter(data)]
                if self.pre_transform is not None:
                    data_list = [self.pre_transform(data) for data in data_list]

                torch.save(self.collate(data_list), self.processed_paths[s + 3 * m])


    @staticmethod
    def label_to_idx(label):
        return Molecule3D.available_properties.index(label)
    def __repr__(self):
        return '{}({})'.format(self.name, len(self))



    def raw_positions(self) -> list:
        # return self.pos
        get_property = lambda i: self.get(i).pos.unsqueeze(0)

        return [get_property(i) for i in tqdm(range(len(self)))]

    def _get_labels(self, i: int, divide_by_atoms: bool) -> torch.Tensor:
        if not divide_by_atoms:
            return self.get(i).labels.unsqueeze(0)
        else:
            return self.get(i).labels.unsqueeze(0) / self.get(i).pos.shape[0]

    def _get_labels_v2(self, divide_by_atoms: bool) -> torch.Tensor:
        if not divide_by_atoms:
            return self.labels
        else:
            return self.labels / self.nodes.unsqueeze(-1).to(torch.float64)

    def _get_labels_ext(self, target: int = None, divide_by_atoms: bool =True, fast=True) -> torch.Tensor:
        print("Trigering")
        if (not fast) or 'nodes' not in self.get(0).keys():
            print("Warning: 'nodes' not in self, using slow method to get labels")
            get_labels = lambda i: self._get_labels(i, divide_by_atoms)
            y = torch.cat([get_labels(i) for i in tqdm(range(len(self)))], dim=0)
        else:
            print("Using fast method to get labels")
            y = self._get_labels_v2(divide_by_atoms)


        if target is None:
            return y[:, :] # .mean(axis=0)
        else:
            return y[:, target] # float(.mean())

    def mean(self, target: int = None, divide_by_atoms=True, fast=True) -> float:
        labels = self._get_labels_ext(target, divide_by_atoms, fast)

        if target is None:
            return labels.mean(axis=0)
        else:
            return float(labels[:, target].mean())
    def std(self, target: int = None, divide_by_atoms=True, fast=True) -> float:
        labels = self._get_labels_ext(target, divide_by_atoms, fast)

        if target is None:
            return labels.std(axis=0)
        else:
            return float(labels[:, target].std())

    def raw_labels(self, target: int = None, divide_by_atoms=True, fast=True) -> float:
        labels = self._get_labels_ext(target, divide_by_atoms, fast)
        return labels


    def atomref(self, target) -> Optional[torch.Tensor]:
        return None
