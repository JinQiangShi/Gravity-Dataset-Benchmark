import os
import torch
from torch.utils.data import DataLoader, Subset, Dataset
from typing import List

from .dataset import ZarrDataset


class ZarrDataloader:
    def __init__(
        self,
        zarr_path: str,
        batch_size: int,
        num_workers: int = 0,
        dtype: torch.dtype = torch.float32,
    ):
        self.dataset = ZarrDataset(zarr_path, dtype=dtype)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.zarr_name = os.path.splitext(os.path.basename(zarr_path))[0]
        self._init_dataset()

    def _init_dataset(self):
        n = len(self.dataset)
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)

        self.train_dataset = Subset(self.dataset, range(0, train_end))
        self.val_dataset = Subset(self.dataset, range(train_end, val_end))
        self.test_dataset = Subset(self.dataset, range(val_end, n))

    @property
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
    
    @property
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
    
    @property
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    @staticmethod
    def merge(
        datasets: List[Dataset],
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 0,
        ) -> DataLoader:
        """
        merge multiple datasets into one dataloader

        Params:
        -----
            datasets (List[Dataset]): list of datasets
        """
        merged_dataset = torch.utils.data.ConcatDataset(datasets)
        return DataLoader(
            merged_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )