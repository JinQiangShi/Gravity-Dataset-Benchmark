import zarr
import torch
from typing import Tuple
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

class ZarrDataset(Dataset):
    def __init__(
        self,
        zarr_path: str,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Params:
        -----
            zarr_path: zarr file path
            dtype: output tensor data type
        """
        zarr_file = zarr.open(zarr_path, mode="r")
        self.density = zarr_file["density"]
        self.gravity = zarr_file["gravity_config1"]
        self.dtype = dtype

        self.density_shape = self.density.shape[-2:] # [nz, nx]

    def __len__(self) -> int:
        return self.density.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.gravity[idx]
        data = torch.from_numpy(data).to(self.dtype)
        data = self.gravity_interpolate(data) # [channels, nx]

        label = self.density[idx]
        label = torch.from_numpy(label).to(self.dtype) # [nz, nx]
        label = label.unsqueeze(0) # [1, nz, nx]

        return data, label

    def gravity_interpolate(self, data: torch.Tensor) -> torch.Tensor:
        """
        interpolate gravity data to match density data
        """
        interpolated_data = F.interpolate(
            data.unsqueeze(0), # [1, channels, nx]
            size=self.density_shape[1], # target nx
            mode="linear", 
            align_corners=True
        ).squeeze(0)
        return interpolated_data

def zarr_dataloader(
    zarr_path: str,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    dtype: torch.dtype = torch.float32,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    split dataset into 6:2:2 ratio

    Params:
    -----
        zarr_path: zarr file path
        batch_size: batch size
        shuffle: shuffle train set data
        num_workers: num of worker number
        dtype: output tensor data type

    Returns:
    --------
        Tuple[train_loader, val_loader, test_loader]
    """
    dataset = ZarrDataset(zarr_path, dtype=dtype)
    n = len(dataset)

    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    train_dataset = Subset(dataset, range(0, train_end))
    val_dataset = Subset(dataset, range(train_end, val_end))
    test_dataset = Subset(dataset, range(val_end, n))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader