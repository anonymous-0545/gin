import logging
import time


import numpy as np
import os
import pytorch_lightning as pl
import shutil
import tempfile
import torch
import torch.nn as nn
from rich.pretty import pprint
from torch.utils.data import random_split, Subset
from torch.utils.data.sampler import RandomSampler
from typing import List, Union
import numpy as np

from torch_geometric.loader import DataLoader

from sainn import utils
from .components.molx import Molecule3D

log = utils.get_logger(__name__)

def cast_to_long(batch):
    batch.z = batch.z.long()
    return batch

class Molecule3DDataModule(pl.LightningDataModule):
    def __init__(
            self,
            batch_size: int,
            data_path: str = './data',
            data_folder: str = "Molecule3Dv2",
            val_batch_size: int = None,
            test_batch_size: int = None,
            split: str = "random",
            num_workers: int = 4,
            pin_memory: bool = True,
            label=None,
            standardize=True,
            divide_by_atoms=True,
    ):
        super().__init__()

        self.divide_by_atoms = divide_by_atoms
        self.standardize = standardize
        self.data_path = data_path
        self.data_folder = data_folder
        self.dataset = None

        self.label = label
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size
        self.test_batch_size = test_batch_size or self.val_batch_size

        assert split in ["random", "scaffold"], "Split must be either random or scaffold"
        self.split = split
        self.data_mean = None
        self.data_std = None
        self.pin_memory = pin_memory
        self.num_workers = num_workers

    @staticmethod
    def dataset_class():
        return Molecule3D

    def get_metadata(self, label_key: Union[str, int] = -1):
        self.label = label_key
        log.info(f"Extract data statistics")
        self.setup(stage="fit")

        self.data_mean = self.data_mean.reshape(1 , -1)
        self.data_std = self.data_std.reshape(1 , -1)

        if not self.standardize:
            self.data_mean = torch.zeros_like(self.data_mean)
            self.data_std = torch.ones_like(self.data_std)


        dataset_metadata = {
            "atomref": self.train_data.atomref(label_key),
            "mean": self.data_mean[0],
            "stddev": self.data_std[0]
        }

        pprint(dataset_metadata)

        return dataset_metadata


    def setup(self, stage=None):
        self.train_data = Molecule3D(self.data_path, split='train', split_mode=self.split, data_folder=self.data_folder,
                                     transform=cast_to_long, label=self.label)


        self.val_data = Molecule3D(self.data_path, split='val', split_mode=self.split, data_folder=self.data_folder,
                                     transform=cast_to_long, label=self.label)
        self.test_data = Molecule3D(self.data_path, split='test', split_mode=self.split, data_folder=self.data_folder,
                                     transform=cast_to_long, label=self.label)

        # self.get_dataset_info()
        processed_data_dir = self.train_data.processed_dir
        log.info(f"Processed data dir: {processed_data_dir}")

        seed = 42
        split_cache_name  = f"cache_divide_t{self.split}_s{seed}.npz"
        split_cache = os.path.join(processed_data_dir, split_cache_name)



        if split_cache is not None and os.path.exists(split_cache):
            S = np.load(split_cache)
            mean =  S["train_mean"]
            std =  S["train_std"]
        else:
            log.info('Cache file is missing generating...')
            split_idx = {}
            mean = self.train_data.mean(divide_by_atoms=self.divide_by_atoms)
            std = self.train_data.std(divide_by_atoms=self.divide_by_atoms)
            split_idx['train_mean'] = mean
            split_idx['train_std'] = std
            log.info(f'Split file is genereated, saving to {split_cache}...')
            np.savez(split_cache, **split_idx)

        self.data_mean = torch.tensor(mean)
        self.data_std = torch.tensor(std)

        log.info(f'train, validation, test: {len(self.train_data)}, {len(self.val_data)}, {len(self.test_data)}')


    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            self.batch_size,
            shuffle=True,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            self.val_batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            self.test_batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers
        )
