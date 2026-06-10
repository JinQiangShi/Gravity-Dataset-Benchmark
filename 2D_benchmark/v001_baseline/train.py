import os
import time
import argparse
import torch

from DATASET_PATH import DATASET_PATH
from dataloader import zarr_dataloader
from model import Gravity_Inverse_UNet_2D
from loss import CombinedLoss
from validate import validate, testify
from utils import TrainingLogger, build_scheduler, set_seed, save_checkpoint

# higher-is-better direction for each candidate metric
METRIC_DIRECTIONS = {"loss": False, "psnr": True, "ssim": True, "mae": False}


def parse_args():
    parser = argparse.ArgumentParser(description="2D Gravity Inverse Training")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--tv_weight", type=float, default=0.01)
    parser.add_argument("--in_channels", type=int, default=2)
    parser.add_argument("--out_channels", type=int, default=126)
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "step", "warmup_cosine", "warmup_step", "none"])
    parser.add_argument("--step_size", type=int, default=30)  # for step scheduler
    parser.add_argument("--gamma", type=float, default=0.1)  # for step scheduler
    parser.add_argument("--warmup_epochs", type=int, default=10,
                        help="number of warmup epochs (for warmup_cosine / warmup_step)")
    parser.add_argument("--save_dir_checkpoints", type=str, default="checkpoints")
    parser.add_argument("--save_dir_test_results", type=str, default="test_results")
    parser.add_argument("--dataset_type", type=str, default="geo_model",
                        choices=["geo_model", "salt_model", "all"])
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=str, default=None, help="checkpoint path to resume from")
    parser.add_argument("--logger", type=str, default="tensorboard",
                        choices=["tensorboard", "swanlab"],
                        help="training tracker: tensorboard or swanlab")
    parser.add_argument("--tensorboard_log_dir", type=str, default="logs",
                        help="tensorboard log directory (used when --logger tensorboard)")
    parser.add_argument("--swanlab_project", type=str, default="Gravity_Dataset_Benchmark_2D",
                        help="swanlab project name (used when --logger swanlab)")
    parser.add_argument("--swanlab_workspace", type=str, default=None,
                        help="swanlab workspace name (used when --logger swanlab)")
    # parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    parser.add_argument("--best_metric", type=str, default="loss",
                        choices=["loss", "psnr", "ssim", "mae"],
                        help="validation metric used to select the best model")
    return parser.parse_args()


def get_dataset_paths(dataset_type: str) -> dict:
    """Collect zarr paths based on dataset type selection."""
    if dataset_type == "all":
        paths = {}
        for category in DATASET_PATH.values():
            paths.update(category)
        return paths
    return DATASET_PATH[dataset_type]


def train_one_epoch(model, loaders, criterion, optimizer, device):
    model.train()
    total_loss, total_batches = 0.0, 0
    for loader in loaders:
        for data, label in loader:
            data, label = data.to(device), label.to(device)
            pred = model(data)
            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
            optimizer.step()
            total_loss += loss.item()
            total_batches += 1
    return total_loss / max(total_batches, 1)


def load_dataloaders(dataset_paths, args):
    """Build train/val/test dataloaders for each dataset."""
    train_loaders, val_loaders, test_loaders, test_names = [], [], [], []
    for name, zarr_path in dataset_paths.items():
        if not os.path.exists(zarr_path):
            print(f"Warning: {zarr_path} not found, skipping.")
            continue
        tr, va, te = zarr_dataloader(zarr_path, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.num_workers)
        train_loaders.append(tr)
        val_loaders.append(va)
        test_loaders.append(te)
        test_names.append(name)
        print(f"  {name}: train={len(tr.dataset)}, val={len(va.dataset)}, test={len(te.dataset)}")
    if not train_loaders:
        raise RuntimeError("No valid datasets found.")
    return train_loaders, val_loaders, test_loaders, test_names


def run_test(model, test_loaders, test_names, criterion, device, results_dir, sub_folder, ckpt_path=None):
    """Run test evaluation, optionally loading from a checkpoint."""
    if ckpt_path and os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"\n--- Test Evaluation (Best Model, epoch {ckpt['epoch'] + 1}) ---")
    else:
        print(f"\n--- Test Evaluation ({sub_folder}) ---")
    loss, psnr, ssim, mae = testify(model, test_loaders, test_names, criterion, device, results_dir, sub_folder)
    print(f"Test Loss: {loss:.6f} | PSNR: {psnr:.2f} | SSIM: {ssim:.4f} | MAE: {mae:.6f}")


def main():
    args = parse_args()
    # set_seed(args.seed)
    set_seed()
    os.makedirs(args.save_dir_checkpoints, exist_ok=True)
    os.makedirs(args.save_dir_test_results, exist_ok=True)

    # Logger (init early to capture full training log)
    logger = TrainingLogger(
        backend=args.logger, log_dir=args.tensorboard_log_dir,
        project=args.swanlab_project, workspace=args.swanlab_workspace, config=vars(args),
    )

    # Device
    device = torch.device(args.device)
    print(f"Device: {device}")

    # Dataloaders
    dataset_paths = get_dataset_paths(args.dataset_type)
    print(f"Datasets: {list(dataset_paths.keys())}")
    train_loaders, val_loaders, test_loaders, test_names = load_dataloaders(dataset_paths, args)

    # Model
    model = Gravity_Inverse_UNet_2D(
        in_channels=args.in_channels, out_channels=args.out_channels,
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss, optimizer, scheduler
    criterion = CombinedLoss(mse_weight=1.0, tv_weight=args.tv_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(
        optimizer, args.scheduler, args.epochs,
        step_size=args.step_size, gamma=args.gamma, warmup_epochs=args.warmup_epochs,
    )

    # Best model selection
    higher_is_better = METRIC_DIRECTIONS[args.best_metric]
    best_value = float("-inf") if higher_is_better else float("inf")
    is_better = lambda new: (new > best_value) if higher_is_better else (new < best_value)

    # Resume
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler and ckpt["scheduler_state_dict"]:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_value = ckpt.get("best_metric_value", best_value)
        print(f"Resumed from epoch {start_epoch}, best_{args.best_metric}={best_value:.6f}")
    print(f"Best-model selection: {args.best_metric} "
          f"({'higher' if higher_is_better else 'lower'} is better)")

    # Training loop
    save_interval = max(1, args.epochs // 5)  # save ~5 checkpoints total
    metrics = {}
    ckpt_dir = args.save_dir_checkpoints

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loaders, criterion, optimizer, device)
        val_loss, val_psnr, val_ssim, val_mae = validate(model, val_loaders, criterion, device)
        if scheduler:
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        print(f"Epoch [{epoch+1}/{args.epochs}] lr={lr:.6f} | "
              f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
              f"val_psnr={val_psnr:.2f} | val_ssim={val_ssim:.4f} | "
              f"val_mae={val_mae:.6f} | time={elapsed:.1f}s")

        # Best model selection
        metrics = {"loss": val_loss, "psnr": val_psnr, "ssim": val_ssim, "mae": val_mae}
        cur = metrics[args.best_metric]
        if is_better(cur):
            best_value = cur
            save_checkpoint(model, optimizer, scheduler, epoch, best_value,
                            os.path.join(ckpt_dir, "best_model.pth"),
                            best_metric_name=args.best_metric, metrics=metrics)
            print(f"  -> New best model ({args.best_metric}={cur:.6f})")

        # Periodic checkpoint (~5 total)
        if (epoch + 1) % save_interval == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, cur,
                            os.path.join(ckpt_dir, f"epoch_{epoch+1}.pth"),
                            best_metric_name=args.best_metric, metrics=metrics)

        # Logger
        logger.log_scalars({
            "Loss/train": train_loss, "Loss/val": val_loss,
            "Metrics/psnr": val_psnr, "Metrics/ssim": val_ssim, "Metrics/mae": val_mae,
            "LR": lr,
        }, step=epoch)

    # Save last model
    save_checkpoint(model, optimizer, scheduler, args.epochs - 1, cur,
                    os.path.join(ckpt_dir, "last_model.pth"),
                    best_metric_name=args.best_metric, metrics=metrics)

    # Test evaluation
    results_dir = args.save_dir_test_results
    run_test(model, test_loaders, test_names, criterion, device, results_dir, "last_model",
             ckpt_path=os.path.join(ckpt_dir, "last_model.pth"))
    run_test(model, test_loaders, test_names, criterion, device, results_dir, "best_model",
             ckpt_path=os.path.join(ckpt_dir, "best_model.pth"))

    logger.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
