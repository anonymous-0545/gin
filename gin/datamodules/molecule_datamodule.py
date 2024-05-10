from os.path import join
from typing import Union

import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_scatter import scatter
from tqdm import tqdm

from .components.md17 import MD17
from .components.qm9 import QM9

def make_splits(dataset_len, train_size, val_size, test_size, seed, filename=None, splits=None):
    if splits is not None:
        splits = np.load(splits)
        idx_train = splits["idx_train"]
        idx_val = splits["idx_val"]
        idx_test = splits["idx_test"]
    else:
        idx_train, idx_val, idx_test = train_val_test_split(dataset_len, train_size, val_size, test_size, seed)

    if filename is not None:
        np.savez(filename, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)

    return torch.from_numpy(idx_train), torch.from_numpy(idx_val), torch.from_numpy(idx_test)



def normalize_positions(batch):
    center = batch.center_of_mass
    batch.pos = batch.pos - center
    return batch

from abc import ABC, abstractmethod

class DatasetPreparer(ABC):
    @abstractmethod
    def prepare(self, datamodule):
        pass

class MD17DatasetPreparer(DatasetPreparer):
    def prepare(self, datamodule):
        datamodule.dataset = MD17(root=datamodule.hparams["dataset_root"], dataset_arg=datamodule.hparams["dataset_arg"])
        train_size = datamodule.hparams["train_size"]
        val_size = datamodule.hparams["val_size"]
        print(datamodule.hparams)
        if "split_type" in datamodule.hparams and datamodule.hparams["split_type"] == "left":
            out = datamodule.dataset.get_idx_split(datamodule.hparams["train_size"], datamodule.hparams["val_size"], datamodule.hparams["seed"])
            idx_train, idx_val, idx_test = out['train'], out['valid'], out['test']
        else:
            idx_train, idx_val, idx_test = make_splits(
                len(datamodule.dataset),
                train_size,
                val_size,
                None,
                datamodule.hparams["seed"],
                join(datamodule.hparams["output_dir"], "splits.npz"),
                datamodule.hparams["splits"],
            )
        return idx_train, idx_val, idx_test


class QM9DatasetPreparer(DatasetPreparer):
    def prepare(self, datamodule):
        transform = normalize_positions if datamodule.hparams["normalize_positions"] is None else None
        if transform:
            print("Normalizing positions.")
        datamodule.dataset = QM9(root=datamodule.hparams["dataset_root"], dataset_arg=datamodule.hparams["dataset_arg"],
                                 transform=transform)
        train_size = datamodule.hparams["train_size"]
        val_size = datamodule.hparams["val_size"]

        idx_train, idx_val, idx_test = make_splits(
            len(datamodule.dataset),
            train_size,
            val_size,
            None,
            datamodule.hparams["seed"],
            join(datamodule.hparams["output_dir"], "splits.npz"),
            datamodule.hparams["splits"],
        )
        return idx_train, idx_val, idx_test


class MoleculeDataModule(LightningDataModule):
    def __init__(self, hparams):
        if type(hparams) == "omegaconf.dictconfig.DictConfig":
            hparams = dict(hparams)
        super(MoleculeDataModule, self).__init__()
        hparams = dict(hparams)

        self.hparams.update(hparams.__dict__) if hasattr(hparams, "__dict__") else self.hparams.update(hparams)
        self._mean, self._std = None, None
        self._saved_dataloaders = dict()
        self.dataset = None

        self.dataset_preparers = {
            'MD17': MD17DatasetPreparer(),
            'QM9': QM9DatasetPreparer()
        }
        self.loaded = False

    def get_metadata(self, label=None):
        if label is not None:
            self.hparams["dataset_arg"] = label

        if self.loaded == False:
            self.prepare_dataset()
            self.loaded = True

        return {
            'atomref': self.atomref,
            'dataset': self.dataset,
            'mean': self.mean,
            'std': self.std
        }


    @staticmethod
    def dataset_class2():
        return QM9

    def prepare_dataset(self):
        dataset_preparer = self.dataset_preparers[self.hparams['dataset']]
        self.idx_train, self.idx_val, self.idx_test = dataset_preparer.prepare(self)

        print(f"train {len(self.idx_train)}, val {len(self.idx_val)}, test {len(self.idx_test)}")
        self.train_dataset = self.dataset[self.idx_train]
        self.val_dataset = self.dataset[self.idx_val]
        self.test_dataset = self.dataset[self.idx_test]

        if self.hparams["standardize"]:
            self._standardize()

    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset, "train")

    def val_dataloader(self):
        loaders = [self._get_dataloader(self.val_dataset, "val")]
        delta = 1 if self.hparams['reload'] == 1 else 2
        if self.hparams["test_interval"] != 0 and (
            len(self.test_dataset) > 0
            and (self.trainer.current_epoch + delta) % self.hparams["test_interval"] == 0
        ):
            loaders.append(self._get_dataloader(self.test_dataset, "test"))
        return loaders

    def test_dataloader(self):
        return self._get_dataloader(self.test_dataset, "test")

    @property
    def atomref(self):
        if hasattr(self.dataset, "get_atomref"):
            return self.dataset.get_atomref()
        return None

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    def _get_dataloader(self, dataset, stage, store_dataloader=True):
        store_dataloader = (store_dataloader and not self.hparams["reload"])
        if stage in self._saved_dataloaders and store_dataloader:
            return self._saved_dataloaders[stage]

        if stage == "train":
            batch_size = self.hparams["batch_size"]
            shuffle = True
        elif stage in ["val", "test"]:
            batch_size = self.hparams["inference_batch_size"]
            shuffle = False

        dl = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.hparams["num_workers"],
            pin_memory=True,
        )

        if store_dataloader:
            self._saved_dataloaders[stage] = dl
        return dl
    
    @rank_zero_only
    def _standardize(self):
        def get_label(batch, atomref):
            if batch.y is None:
                raise MissingLabelException()

            dy = None
            if 'dy' in batch:
                dy = batch.dy.squeeze().clone()

            if atomref is None:
                return batch.y.clone(), dy

            atomref_energy = scatter(atomref[batch.z], batch.batch, dim=0)
            return (batch.y.squeeze() - atomref_energy.squeeze()).clone(), dy

        if self.hparams["standardize"] == 2:
            label = self.hparams["dataset_arg"]
            print(
                f" standardization is used for label {label}."
            )

            self._mean = self.train_dataset.mean()
            self._std = self.train_dataset.std()
            return None

        data = tqdm(
            self._get_dataloader(self.train_dataset, "val", store_dataloader=False), 
            desc="computing mean and std",
        )
        try:
            atomref = self.atomref if self.hparams["prior_model"] == "Atomref" else None
            ys = [get_label(batch, atomref) for batch in data]
            # convert array with n elements and each element cotains 2 value to array of two elements with n values
            ys, dys = zip(*ys)
            ys = torch.cat(ys)
        except MissingLabelException:
            rank_zero_warn(
                "Standardize is true but failed to compute dataset mean and "
                "standard deviation. Maybe the dataset only contains forces."
            )
            return None

        self._mean = ys.mean(dim=0)[0].item()
        self._std = ys.std(dim=0)[0].item()
        print(f"mean: {self._mean}, std: {self._std}")

