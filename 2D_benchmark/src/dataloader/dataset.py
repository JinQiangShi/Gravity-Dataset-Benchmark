import zarr
import torch
from typing import Tuple
import torch.nn.functional as F
from torch.utils.data import Dataset


class ZarrDataset(Dataset):
    def __init__(
        self,
        zarr_path: str,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Parameters:
        -----------
            zarr_path (str): zarr file path
            dtype (torch.dtype): output tensor data type
        """
        self.root = zarr.open(zarr_path, mode="r")
        self.model = self.root["density"] # density model, [batch, model_nz, model_nx]
        self.data = self.root["gravity1"] # gravity data, [batch, channels, data_nx]
        self.dtype = dtype
        self.length = self.model.shape[0]
        self.model_nz, self.model_nx = self.model.shape[-2:]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.data[idx] # np.float64, [channels, data_nx]
        data = torch.from_numpy(data).to(self.dtype)
        data = self.gravity_interpolate(data) # [channels, model_nx]
        data_gradient = self.gravity_gradient(data) # [channels, model_nx]
        data = torch.cat([data, data_gradient], dim=0) # [2*channels, model_nx]

        model = self.model[idx] # np.int8, [model_nz, model_nx]
        model = torch.from_numpy(model).to(self.dtype)
        model = model.unsqueeze(0) # [1, model_nz, model_nx]

        return data, model

    def gravity_interpolate(self, data: torch.Tensor) -> torch.Tensor:
        """
        interpolate gravity data to match density model
        """
        interpolated_data = F.interpolate(
            data.unsqueeze(0), # [1, channels, data_nx]
            size=self.model_nx,
            mode="linear", 
            align_corners=True # make endpoints aligned
        ).squeeze(0)
        return interpolated_data

    def gravity_gradient(self, data: torch.Tensor) -> torch.Tensor:
        """
        calculate gradient of gravity data
        """
        gradient = torch.diff(data, dim=1) # [channels, data_nx-1]
        gradient = self.gravity_interpolate(gradient) # [channels, model_nx]
        return gradient

