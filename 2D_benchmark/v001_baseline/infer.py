import os
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from DATASET_PATH import DATASET_PATH
from dataloader import zarr_dataloader
from model import Gravity_Inverse_UNet_2D
from loss import CombinedLoss
from validate import testify
from utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="2D Gravity Inverse Inference")
    parser.add_argument("--checkpoint", type=str, nargs="+", required=True,
                        help="path(s) to trained model checkpoint(s) (.pth)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--in_channels", type=int, default=2)
    parser.add_argument("--out_channels", type=int, default=126)
    parser.add_argument("--tv_weight", type=float, default=0.01)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="test_results",
                        help="directory to save prediction files")
    parser.add_argument("--sub_folder", type=str, nargs="+", default=None,
                        help="output sub-folder name(s), one per checkpoint (default: checkpoint filename stem)")
    parser.add_argument("--save_format", type=str, default="npz",
                        choices=["npz", "zarr"],
                        help="output file format (default: npz)")
    # zarr dataset mode
    parser.add_argument("--dataset_type", type=str, default="geo_model",
                        choices=["geo_model", "salt_model", "all"])
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"])
    # custom numpy mode
    parser.add_argument("--input", type=str, default=None,
                        help="path to .npy/.npz file or a directory of them")
    parser.add_argument("--target_nx", type=int, default=None,
                        help="gravity interpolation target nx (defaults to --out_channels)")
    parser.add_argument("--input_key", type=str, default="gravity",
                        help="key name in .npz file (ignored for .npy)")
    return parser.parse_args()


class CustomGravityDataset(Dataset):
    """Load custom gravity data from .npy/.npz for inference."""

    def __init__(self, path, target_nx, input_key="gravity", dtype=torch.float32):
        self.target_nx = target_nx
        self.dtype = dtype

        raw = np.load(path)
        arr = raw[input_key] if path.lower().endswith(".npz") else raw
        arr = arr.astype(np.float32)
        if arr.ndim == 2:
            arr = arr[None]
        elif arr.ndim != 3:
            raise ValueError(f"Expected 2D or 3D array, got shape {arr.shape}")
        self.data = torch.from_numpy(arr).to(dtype)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self._interpolate(self.data[idx]), torch.zeros(1)

    def _interpolate(self, data):
        """same usage like ZarrDataset.gravity_interpolate"""
        return F.interpolate(
            data.unsqueeze(0), size=self.target_nx,
            mode="linear", align_corners=True,
        ).squeeze(0)


def load_model(checkpoint, device, in_channels=2, out_channels=126):
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model = Gravity_Inverse_UNet_2D(
        in_channels=in_channels, out_channels=out_channels,
    ).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    epoch = ckpt.get("epoch", "?")
    best = ckpt.get("best_metric_value")
    best_name = ckpt.get("best_metric_name")
    extra = f", best {best_name}: {best:.6f}" if best is not None else ""
    print(f"Loaded checkpoint: {checkpoint}\n  Epoch: {epoch+1}{extra}")
    return model


def collect_numpy_files(input_path):
    if os.path.isfile(input_path):
        return [input_path]
    if os.path.isdir(input_path):
        files = sorted(
            os.path.join(input_path, f) for f in os.listdir(input_path)
            if f.lower().endswith((".npy", ".npz"))
        )
        if not files:
            raise RuntimeError(f"No .npy/.npz files found in {input_path}")
        return files
    raise FileNotFoundError(f"Input not found: {input_path}")


def _save_custom(path, data, pred, fmt="npz"):
    """Save inference arrays in the specified format."""
    if fmt == "npz":
        np.savez(path, data=data, pred=pred)
    elif fmt == "zarr":
        import zarr
        store = zarr.storage.LocalStore(path)
        root = zarr.open_group(store, mode="w")
        batch = 10
        root.create_array("data", data=data, chunks=(batch, *data.shape[1:]))
        root.create_array("pred", data=pred, chunks=(batch, *pred.shape[1:]))
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def _run_batches(loader, model, device):
    """Run inference over a DataLoader and return concatenated data/pred arrays."""
    preds, datas = [], []
    for data, _ in loader:
        data = data.to(device)
        preds.append(model(data).cpu().numpy())
        datas.append(data.cpu().numpy())
    return np.concatenate(datas, axis=0), np.concatenate(preds, axis=0)


def infer_custom(model, args, device):
    """Run inference on custom numpy gravity files."""
    target_nx = args.target_nx or args.out_channels
    files = collect_numpy_files(args.input)
    save_dir = os.path.join(args.save_dir, args.sub_folder)
    os.makedirs(save_dir, exist_ok=True)
    ext = f".{args.save_format}"

    print(f"\n--- Custom Inference ({len(files)} file(s), target_nx={target_nx}) ---")
    model.eval()
    with torch.no_grad():
        for fpath in files:
            ds = CustomGravityDataset(fpath, target_nx, input_key=args.input_key)
            loader = DataLoader(ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers)
            data_arr, pred_arr = _run_batches(loader, model, device)

            stem = os.path.splitext(os.path.basename(fpath))[0]
            out_path = os.path.join(save_dir, f"{stem}_pred{ext}")
            _save_custom(out_path, data_arr, pred_arr, fmt=args.save_format)
            print(f"  {fpath} -> {out_path}  (N={pred_arr.shape[0]})")


def _resolve_dataset_paths(dataset_type):
    if dataset_type == "all":
        paths = {}
        for category in DATASET_PATH.values():
            paths.update(category)
        return paths
    return DATASET_PATH[dataset_type]


def infer_dataset(model, args, device):
    """Run inference on registered zarr datasets."""
    dataset_paths = _resolve_dataset_paths(args.dataset_type)
    print(f"Datasets: {list(dataset_paths.keys())}")

    loaders, loader_names = [], []
    split_idx = {"train": 0, "val": 1, "test": 2}[args.split]
    for name, zarr_path in dataset_paths.items():
        if not os.path.exists(zarr_path):
            print(f"Warning: {zarr_path} not found, skipping.")
            continue
        loaders_split = zarr_dataloader(
            zarr_path, batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers,
        )
        loaders.append(loaders_split[split_idx])
        loader_names.append(name)
        print(f"  {name}: {args.split}={len(loaders[-1].dataset)}")

    if not loaders:
        raise RuntimeError("No valid datasets found.")

    criterion = CombinedLoss(mse_weight=1.0, tv_weight=args.tv_weight)
    print(f"\n--- Inference ({args.split}) ---")
    loss, psnr_val, ssim_val, mae_val = testify(
        model, loaders, loader_names, criterion,
        device, args.save_dir, args.sub_folder, save_fmt=args.save_format,
    )
    print(f"Loss: {loss:.6f} | PSNR: {psnr_val:.2f} | "
          f"SSIM: {ssim_val:.4f} | MAE: {mae_val:.6f}")
    print(f"Results saved to: {os.path.join(args.save_dir, args.sub_folder)}")


def main():
    args = parse_args()
    set_seed()
    os.makedirs(args.save_dir, exist_ok=True)

    checkpoints = args.checkpoint
    sub_folders = args.sub_folder
    if sub_folders is not None and len(sub_folders) != len(checkpoints):
        raise ValueError(
            f"Number of --sub_folder ({len(sub_folders)}) must match "
            f"--checkpoint ({len(checkpoints)})"
        )

    device = torch.device(args.device)
    print(f"Device: {device}")

    for i, ckpt in enumerate(checkpoints):
        sub = sub_folders[i] if sub_folders else os.path.splitext(os.path.basename(ckpt))[0]
        args.checkpoint = ckpt
        args.sub_folder = sub
        print(f"\n=== [{i + 1}/{len(checkpoints)}] checkpoint: {ckpt}  sub_folder: {sub} ===")

        model = load_model(ckpt, device, args.in_channels, args.out_channels)

        if args.input is not None:
            infer_custom(model, args, device)
        else:
            infer_dataset(model, args, device)


if __name__ == "__main__":
    main()
