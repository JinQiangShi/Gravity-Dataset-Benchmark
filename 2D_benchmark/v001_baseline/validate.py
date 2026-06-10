import os
import numpy as np
import torch

from evaluation import psnr, ssim, mae


def _save_npz(path, data, label, pred):
    """Save arrays as an .npz file."""
    np.savez(path, data=data, label=label, pred=pred)


def _save_zarr(path, data, label, pred):
    """Save arrays as a .zarr file (zarr v3 API)."""
    import zarr
    store = zarr.storage.LocalStore(path)
    root = zarr.open_group(store, mode="w")
    batch = 10
    root.create_array("data", data=data, chunks=(batch, *data.shape[1:]))
    root.create_array("label", data=label, chunks=(batch, *label.shape[1:]))
    root.create_array("pred", data=pred, chunks=(batch, *pred.shape[1:]))


_SAVERS = {
    "npz": (_save_npz, ".npz"),
    "zarr": (_save_zarr, ".zarr"),
}


def save_results(save_dir, sub_folder, name, data, label, pred, fmt="npz"):
    """Save evaluation result arrays to disk.

    Args:
        save_dir: root output directory.
        sub_folder: sub-folder name under *save_dir*.
        name: original loader name (extension is stripped).
        data, label, pred: numpy arrays to persist.
        fmt: output format key, one of ``_SAVERS`` (default ``"npz"``).
             Add new formats by registering a ``(saver_fn, ext)`` pair in
             ``_SAVERS``.
    """
    if fmt not in _SAVERS:
        raise ValueError(f"Unsupported save format '{fmt}'. Choose from {list(_SAVERS)}")
    saver_fn, ext = _SAVERS[fmt]
    out_path = os.path.join(save_dir, sub_folder, f"{name.split('.')[0]}{ext}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    saver_fn(out_path, data, label, pred)
    print(f"  Saved: {out_path}  (N={len(data)})")


def _evaluate_loop(model, loaders, criterion, device,
                   loader_names=None, save_dir=None, sub_folder=None,
                   save_fmt="npz"):
    """Shared evaluation loop for validation and testing.

    Args:
        model: the model to evaluate.
        loaders: list of DataLoaders.
        criterion: loss function.
        device: torch device.
        loader_names: optional list of names per loader (required for saving).
        save_dir: directory to save predictions (required for saving).
        sub_folder: sub-folder name for saved files (required for saving).
        save_fmt: output format key passed to ``save_results`` (default ``"npz"``).

    Returns:
        (avg_loss, avg_psnr, avg_ssim, avg_mae)
    """
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_mae = 0.0
    total_batches = 0

    names = loader_names if loader_names is not None else [None] * len(loaders)
    do_save = save_dir is not None and sub_folder is not None

    for name, loader in zip(names, loaders):
        if do_save:
            all_data, all_label, all_pred = [], [], []

        for data, label in loader:
            data = data.to(device)
            label = label.to(device)

            pred = model(data)
            loss = criterion(pred, label)

            total_loss += loss.item()
            total_psnr += psnr(pred, label).item()
            total_ssim += ssim(pred, label).item()
            total_mae += mae(pred, label).item()
            total_batches += 1

            if do_save:
                all_data.append(data.cpu().numpy())
                all_label.append(label.cpu().numpy())
                all_pred.append(pred.cpu().numpy())

        if do_save:
            data_arr = np.concatenate(all_data, axis=0)
            label_arr = np.concatenate(all_label, axis=0)
            pred_arr = np.concatenate(all_pred, axis=0)
            save_results(save_dir, sub_folder, name, data_arr, label_arr, pred_arr, fmt=save_fmt)

    n = max(total_batches, 1)
    return total_loss / n, total_psnr / n, total_ssim / n, total_mae / n


@torch.no_grad()
def validate(model, loaders, criterion, device):
    """Run validation and return average metrics."""
    return _evaluate_loop(model, loaders, criterion, device)


@torch.no_grad()
def testify(model, loaders, loader_names, criterion, device, save_dir, sub_folder,
            save_fmt="npz"):
    """Compute test metrics and save per-loader predictions."""
    return _evaluate_loop(
        model, loaders, criterion, device,
        loader_names=loader_names, save_dir=save_dir, sub_folder=sub_folder,
        save_fmt=save_fmt,
    )
