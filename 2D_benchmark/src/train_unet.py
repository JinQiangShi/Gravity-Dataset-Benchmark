import os
import torch
import numpy as np
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from typing import List

from .utils import set_seed, Logger
from .dataloader import dataset_path, ZarrDataloader
from .model import get_model, save_checkpoint, load_checkpoint
from .loss import CombinedLoss
from .evaluate import CombinedMetric
from .scheduler import WarmupCosineScheduler

def train_unet(
    save_dir: str,
    device: str,
    max_epoch: int = 200,
    batch_size: int = 32,
    num_workers: int = 0,
    learning_rate: float = 1e-3,
    model_kwargs: dict = {"in_channels": 2*2, "out_channels": 128, "linear": False},
    model_checkpoint: str = None,
    checkpoint_save_interval: int = 20,
    earlystop_patience: int = 20,
):
    """
    Main function for training the model.

    Parameters
    ----------
    save_dir (str): directory to save the model checkpoints and test results.
    device (str): device to use for training.
    max_epoch (int): maximum number of epochs to train, default is 200.
    batch_size (int): batch size for dataloader, default is 32.
    num_workers (int): number of workers for dataloader, default is 0.
    learning_rate (float): learning rate for optimizer, default is 1e-3.
    model_kwargs (dict): keyword arguments for the model, default is UNet kwargs.
    model_checkpoint (str): path to the model checkpoint to load, default is None.
    checkpoint_save_interval (int): interval (in epochs) to save periodic checkpoints, default is 20.
    earlystop_patience (int): number of epochs without improvement before early stopping, default is 20.
    """
    set_seed()

    # save directory
    save_dir_log = os.path.join(save_dir, "log")
    save_dir_checkpoints = os.path.join(save_dir, "checkpoint")
    save_dir_test_results = os.path.join(save_dir, "test_result")
    os.makedirs(save_dir_log, exist_ok=True)
    os.makedirs(save_dir_checkpoints, exist_ok=True)
    os.makedirs(save_dir_test_results, exist_ok=True)

    # logger
    logger = Logger(
        log_dir=save_dir_log,
        project="Gravity_Dataset_Benchmark", 
        workspace="sjq"
    )

    # device
    device = torch.device(device)
    print(f"Using device: {device}")

    # dataloaders
    dataloaders = [
        ZarrDataloader(
            zarr_path = zarr_path,
            batch_size = batch_size,
            num_workers = num_workers,
        ) for _, zarr_path in dataset_path("geo_model").items()
    ]

    train_loader = ZarrDataloader.merge(
        datasets = [dataloader.train_dataset for dataloader in dataloaders],
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers,
    )

    val_loader = ZarrDataloader.merge(
        datasets = [dataloader.val_dataset for dataloader in dataloaders],
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers,
    )

    # model
    model = get_model("unet", model_kwargs)
    model.to(device)
    if model_checkpoint is not None:
        load_checkpoint(
            checkpoint_path = model_checkpoint,
            model = model,
            device = device,
        )

    # loss
    train_criterion = CombinedLoss(
        huber_weight=1.0,
        depth_weight=1.0, 
        ssim_weight=0.03,
        tv_weight=0.03,
        ms_weight=0.03,
        mgs_weight=0.03,
        data_range=1.0,
    )

    # metric
    val_metric = CombinedMetric(
        mae_weight=1.0,
        psnr_weight=0.5,
        ssim_weight=0.3,
        psnr_max=20.0,
        data_range=1.0,
    )
    test_metric = val_metric.copy()

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # scheduler
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=20)

    # loop
    epochs_without_improvement = 0
    for epoch in range(1, max_epoch+1):
        print(f"Epoch {epoch}")

        # =============== training ==============
        _train_one_epoch(
            model = model, 
            train_loader = train_loader,
            optimizer = optimizer,
            criterion = train_criterion,
            device = device
        )
        logger.log_scalars({"lr": optimizer.param_groups[0]["lr"]}, epoch)
        scheduler.step()
        logger.log_scalars(train_criterion.value_dict("train"), epoch)
        train_criterion.reset()

        # =============== validation ==============
        _validate(
            model = model,
            val_loader = val_loader,
            metric = val_metric,
            device = device,
        )
        logger.log_scalars(val_metric.value_dict("val"), epoch)

        # model selection
        # early stopping on the validation metric
        if val_metric.is_better():
            if val_metric.best_metric_value is None:
                print(f"First validation metric: {val_metric.metric():.6f}, saving best checkpoint")
            else:
                print(f"Validation metric improved from {val_metric.best_metric_value:.6f} to {val_metric.metric():.6f}, saving best checkpoint")
            val_metric.update_best()
            save_checkpoint(
                epoch = epoch,
                model = model,
                metric_value_dict = val_metric.value_dict("val"),
                path = os.path.join(save_dir_checkpoints, "best.pth"),
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # periodic checkpoint
        if epoch % checkpoint_save_interval == 0:
            print(f"Saving checkpoint at epoch {epoch}")
            save_checkpoint(
                epoch = epoch,
                model = model,
                metric_value_dict = val_metric.value_dict("val"),
                path = os.path.join(save_dir_checkpoints, f"epoch_{epoch:03d}.pth"),
            )

        # save the final model
        is_last_epoch: bool = epoch == max_epoch
        should_early_stop: bool = epochs_without_improvement >= earlystop_patience
        if is_last_epoch or should_early_stop:
            print(f"Saving final checkpoint at epoch {epoch}")
            save_checkpoint(
                epoch = epoch,
                model = model,
                metric_value_dict = val_metric.value_dict("val"),
                path = os.path.join(save_dir_checkpoints, "final.pth"),
            )

        val_metric.reset()

        if should_early_stop:
            print(f"Early stopping at epoch {epoch}, best metric {val_metric.best_metric_value:.6f}")
            break

    # test with the best checkpoint
    load_checkpoint(
        checkpoint_path = os.path.join(save_dir_checkpoints, "best.pth"),
        model = model,
        device = device,
    )
    _testify(
        model = model,
        zarr_dataloaders = dataloaders,
        metric = test_metric,
        save_dir = save_dir_test_results,
        device = device,
        logger = logger,
    )

    # visualize
    # Commented out because the visualization library causes an OpenMP library conflict 
    # Need to run the visualization script separately

def _train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: CombinedLoss,
    device: str,
) -> None:
    model.train()
    for gravity, density in tqdm(train_loader, desc="train"):
        gravity = gravity.to(device)
        density = density.to(device)
        density_pred = model(gravity)
        loss = criterion(density_pred, density)
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
        optimizer.step()


def _validate(
    model: nn.Module,
    val_loader: DataLoader,
    metric: CombinedMetric,
    device: str,
) -> None:
    model.eval()
    with torch.no_grad():
        for gravity, density in tqdm(val_loader, desc="val"):
            gravity = gravity.to(device)
            density = density.to(device)
            density_pred = model(gravity)
            metric(density_pred, density)

def _testify(
    model: nn.Module,
    zarr_dataloaders: List[ZarrDataloader],
    metric: CombinedMetric,
    save_dir: str,
    device: str,
    logger: Logger,
    max_vis: int = 100,
) -> dict:
    model.eval()
    with torch.no_grad():
        for zarr_dataloader in zarr_dataloaders:
            zarr_name = zarr_dataloader.zarr_name
            test_loader = zarr_dataloader.test_dataloader

            metric.reset()

            save_gravity = []
            save_density = []
            save_pred = []
            saved = 0
            for gravity, density in tqdm(test_loader, desc=f"test {zarr_name}"):
                gravity = gravity.to(device)
                density = density.to(device)
                density_pred = model(gravity)
                metric(density_pred, density)

                if saved < max_vis:
                    take = min(max_vis - saved, gravity.shape[0])
                    save_gravity.append(gravity[:take].cpu().numpy())
                    save_density.append(density[:take].cpu().numpy())
                    save_pred.append(density_pred[:take].cpu().numpy())
                    saved += take

            np.savez(
                os.path.join(save_dir, f"{zarr_name}.npz"),
                gravity = np.concatenate(save_gravity),
                density = np.concatenate(save_density),
                density_pred = np.concatenate(save_pred),
            )
            logger.log_scalars(metric.value_dict(f"{zarr_name}"), 0)
