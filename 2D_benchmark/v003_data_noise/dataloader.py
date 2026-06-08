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
        add_noise: bool = False,
    ):
        """
        Params:
        -----
            zarr_path: zarr file path
            dtype: output tensor data type
            add_noise: whether to add noise to gravity data (only for training)
        """
        zarr_file = zarr.open(zarr_path, mode="r")
        self.density = zarr_file["density"]
        self.gravity = zarr_file["gravity_config1"]
        self.dtype = dtype
        self.add_noise = add_noise

        self.density_shape = self.density.shape[-2:] # [nz, nx]

    def __len__(self) -> int:
        return self.density.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.gravity[idx]
        data = torch.from_numpy(data).to(self.dtype)
        data = self.gravity_interpolate(data) # [channels, nx]
        # data = self.gravity_normalize(data)
        if self.add_noise:
            data = self.gravity_add_noise(data)

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

    # def gravity_normalize(self, data: torch.Tensor) -> torch.Tensor:
    #     """
    #     normalize gravity data: d_norm = (d - mean) / std
    #     """
    #     mean = data.mean()
    #     std = data.std()
    #     return (data - mean) / std

    def gravity_add_noise(self, data: torch.Tensor, max_noise: float = 0.05) -> torch.Tensor:
        """
        add 0%~max_noise random Gaussian noise to gravity data
        """
        noise_level = torch.rand(1, device=data.device) * max_noise
        noise = torch.randn_like(data) * noise_level * data.std()
        return data + noise

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
        (train_loader, val_loader, test_loader)
    """
    train_dataset_full = ZarrDataset(zarr_path, dtype=dtype, add_noise=True)
    eval_dataset_full = ZarrDataset(zarr_path, dtype=dtype, add_noise=False)
    n = len(train_dataset_full)

    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    train_dataset = Subset(train_dataset_full, range(0, train_end))
    val_dataset = Subset(eval_dataset_full, range(train_end, val_end))
    test_dataset = Subset(eval_dataset_full, range(val_end, n))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

def get_test_loader(
    zarr_path: str,
    batch_size: int,
    num_workers: int = 0,
    dtype: torch.dtype = torch.float32,
) -> DataLoader:
    """
    get test dataloader from zarr file

    Params:
    -----
        zarr_path: zarr file path
        batch_size: batch size
        num_workers: num of worker number
        dtype: output tensor data type

    Returns:
    --------
        test_loader
    """
    train_loader, val_loader, test_loader = zarr_dataloader(
        zarr_path,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        dtype=dtype,
    )
    return test_loader
