import os
import numpy as np
import torch

from evaluation import psnr, ssim, mae


def _evaluate_loop(model, loaders, criterion, device,
                   loader_names=None, save_dir=None, tag=None):
    """Shared evaluation loop for validation and testing.

    Args:
        model: the model to evaluate.
        loaders: list of DataLoaders.
        criterion: loss function.
        device: torch device.
        loader_names: optional list of names per loader (required for saving).
        save_dir: directory to save npz predictions (required for saving).
        tag: filename tag for saved npz files (required for saving).

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
    do_save = save_dir is not None and tag is not None

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
            npz_path = os.path.join(save_dir, f"{tag}", f"{name.split('.')[0]}.npz")
            os.makedirs(os.path.dirname(npz_path), exist_ok=True)
            np.savez(npz_path, data=data_arr, label=label_arr, pred=pred_arr)
            print(f"  Saved: {npz_path}  (N={len(data_arr)})")

    n = max(total_batches, 1)
    return total_loss / n, total_psnr / n, total_ssim / n, total_mae / n


@torch.no_grad()
def validate(model, loaders, criterion, device):
    """Run validation and return average metrics."""
    return _evaluate_loop(model, loaders, criterion, device)


@torch.no_grad()
def testify(model, loaders, loader_names, criterion, device, save_dir, tag):
    """Compute test metrics and save per-loader predictions as npz files."""
    return _evaluate_loop(
        model, loaders, criterion, device,
        loader_names=loader_names, save_dir=save_dir, tag=tag,
    )
