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
        augment: bool = False,
    ):
        """
        Params:
        -----
            zarr_path: zarr file path
            dtype: output tensor data type
            augment: whether to apply left-right flip augmentation
        """
        zarr_file = zarr.open(zarr_path, mode="r")
        self.density = zarr_file["density"]
        self.gravity = zarr_file["gravity_config1"]
        self.dtype = dtype
        self.augment = augment

        self.density_shape = self.density.shape[-2:] # [nz, nx]

    def __len__(self) -> int:
        return self.density.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.gravity[idx]
        data = torch.from_numpy(data).to(self.dtype)
        data = self.gravity_interpolate(data) # [channels, nx]
        # data = self.gravity_normalize(data) # [channels, nx]
        # data = self.gravity_add_noise(data) # [channels, nx]

        label = self.density[idx]
        label = torch.from_numpy(label).to(self.dtype) # [nz, nx]
        label = label.unsqueeze(0) # [1, nz, nx]

        # left-right flip augmentation (along nx axis)
        if self.augment and torch.rand(1).item() > 0.5:
            data = torch.flip(data, dims=[-1])
            label = torch.flip(label, dims=[-1])

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
    #     return normalized gravity data
    #     """
    #     # z-score normalize
    #     # return (data - data.mean()) / (data.std() + 1e-8)
        
    #     # min-max normalize
    #     # return (data - data.min()) / (data.max() - data.min() + 1e-8)

    # def gravity_add_noise(self, data: torch.Tensor) -> torch.Tensor:
    #     """
    #     add 5% Gaussian noise to gravity data (noise std = 5% of data std)
    #     """
    #     noise = torch.randn_like(data) * 0.05 * data.std()
    #     return data + noise

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
    train_base = ZarrDataset(zarr_path, dtype=dtype, augment=True)
    val_base = ZarrDataset(zarr_path, dtype=dtype, augment=False)

    n = len(train_base)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    train_dataset = Subset(train_base, range(0, train_end))
    val_dataset = Subset(val_base, range(train_end, val_end))
    test_dataset = Subset(val_base, range(val_end, n))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader