import os
import time
import argparse
import torch

from DATASET_PATH import DATASET_PATH
from dataloader import zarr_dataloader
from model import Gravity_Inverse_UNet_2D
from loss import CombinedLoss
from validate import validate, testify
from utils import TrainingLogger, build_scheduler
from saver import save_checkpoint

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
    parser.add_argument("--step_size", type=int, default=30) # for step scheduler
    parser.add_argument("--gamma", type=float, default=0.1) # for step scheduler
    parser.add_argument("--warmup_epochs", type=int, default=10, help="number of warmup epochs (for warmup_cosine / warmup_step)")
    parser.add_argument("--save_dir_checkpoints", type=str, default="checkpoints")
    parser.add_argument("--save_dir_test_results", type=str, default="test_results")
    parser.add_argument("--dataset_type", type=str, default="geo_model", choices=["geo_model", "salt_model", "all"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=str, default=None, help="checkpoint path to resume from")
    parser.add_argument("--logger", type=str, default="tensorboard", choices=["tensorboard", "swanlab"],
                        help="training tracker: tensorboard or swanlab")
    parser.add_argument("--tensorboard_log_dir", type=str, default="logs",
                        help="tensorboard log directory (used when --logger tensorboard)")
    parser.add_argument("--swanlab_project", type=str, default="Gravity_Dataset_Benchmark_2D",
                        help="swanlab project name (used when --logger swanlab)")
    parser.add_argument("--swanlab_workspace", type=str, default=None,
                        help="swanlab workspace name (used when --logger swanlab)")
    parser.add_argument("--best_metric", type=str, default="loss",
                        choices=["loss", "psnr", "ssim", "mae"],
                        help="validation metric used to select the best model")
    return parser.parse_args()


# higher-is-better direction for each candidate metric
METRIC_DIRECTIONS = {"loss": False, "psnr": True, "ssim": True, "mae": False}


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
    total_loss = 0.0
    total_batches = 0

    for loader in loaders:
        for data, label in loader:
            data = data.to(device)
            label = label.to(device)

            pred = model(data)
            loss = criterion(pred, label)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

    return total_loss / max(total_batches, 1)


def main():
    args = parse_args()
    os.makedirs(args.save_dir_checkpoints, exist_ok=True)
    os.makedirs(args.save_dir_test_results, exist_ok=True)

    # Logger (init early to capture full training log)
    logger = TrainingLogger(
        backend=args.logger,
        log_dir=args.tensorboard_log_dir,
        project=args.swanlab_project,
        workspace=args.swanlab_workspace,
        config=vars(args),
    )
    
    # Device
    device = torch.device(args.device)
    print(f"Device: {device}")

    # Collect dataset paths
    dataset_paths = get_dataset_paths(args.dataset_type)
    print(f"Datasets: {list(dataset_paths.keys())}")

    # Build dataloaders for each dataset
    train_loaders = []
    val_loaders = []
    test_loaders = []
    test_loader_names = []
    for name, zarr_path in dataset_paths.items():
        if not os.path.exists(zarr_path):
            print(f"Warning: {zarr_path} not found, skipping.")
            continue
        train_loader, val_loader, test_loader = zarr_dataloader(
            zarr_path,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        train_loaders.append(train_loader)
        val_loaders.append(val_loader)
        test_loaders.append(test_loader)
        test_loader_names.append(name)
        print(f"  {name}: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, test={len(test_loader.dataset)}")

    if not train_loaders:
        raise RuntimeError("No valid datasets found.")

    # Model
    model = Gravity_Inverse_UNet_2D(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Loss, optimizer, scheduler
    criterion = CombinedLoss(mse_weight=1.0, tv_weight=args.tv_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(
        optimizer, args.scheduler, args.epochs,
        step_size=args.step_size, gamma=args.gamma, warmup_epochs=args.warmup_epochs,
    )

    # Best model selection
    best_metric_name = args.best_metric
    higher_is_better = METRIC_DIRECTIONS[best_metric_name]
    best_metric_value = float("-inf") if higher_is_better else float("inf")

    def is_better(new, old):
        return new > old if higher_is_better else new < old

    # Resume
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler and ckpt["scheduler_state_dict"]:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_metric_value = ckpt.get("best_metric_value", best_metric_value)
        print(f"Resumed from epoch {start_epoch}, best_{best_metric_name}={best_metric_value:.6f}")
    print(f"Best-model selection: {best_metric_name} ({'higher' if higher_is_better else 'lower'} is better)")

    # Training loop
    save_interval = max(1, args.epochs // 5)  # save ~5 checkpoints total
    current_metrics = {}
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loaders, criterion, optimizer, device)
        val_loss, val_psnr, val_ssim, val_mae = validate(model, val_loaders, criterion, device)

        if scheduler:
            scheduler.step()

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch+1}/{args.epochs}] "
            f"lr={current_lr:.6f} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"val_psnr={val_psnr:.2f} | "
            f"val_ssim={val_ssim:.4f} | "
            f"val_mae={val_mae:.6f} | "
            f"time={elapsed:.1f}s"
        )

        # Best model selection
        current_metrics = {"psnr": val_psnr, "ssim": val_ssim, "mae": val_mae, "loss": val_loss}
        cur_value = current_metrics[best_metric_name]
        if is_better(cur_value, best_metric_value):
            best_metric_value = cur_value
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_metric_value,
                os.path.join(args.save_dir_checkpoints, "best_model.pth"),
                best_metric_name=best_metric_name, metrics=current_metrics,
            )
            print(
                f"  -> New best model saved ({best_metric_name}={cur_value:.6f}, "
                f"psnr={val_psnr:.2f}, ssim={val_ssim:.4f}, mae={val_mae:.6f})"
            )

        # Periodic checkpoint, only save ~5 models
        if (epoch + 1) % save_interval == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, cur_value,
                            os.path.join(args.save_dir_checkpoints, f"epoch_{epoch+1}.pth"),
                            best_metric_name=best_metric_name, metrics=current_metrics)

        # Logger
        logger.log_scalars({
            "Loss/train": train_loss,
            "Loss/val": val_loss,
            "Metrics/psnr": val_psnr,
            "Metrics/ssim": val_ssim,
            "Metrics/mae": val_mae,
            "LR": current_lr,
        }, step=epoch)

    # Save final model
    save_checkpoint(model, optimizer, scheduler, args.epochs - 1, cur_value,
                    os.path.join(args.save_dir_checkpoints, "last_model.pth"),
                    best_metric_name=best_metric_name, metrics=current_metrics)

    test_results_dir = args.save_dir_test_results

    # Final test evaluation on last model
    print("\n--- Test Evaluation (Last Model) ---")
    test_loss, test_psnr, test_ssim, test_mae = testify(
        model, test_loaders, test_loader_names, criterion, device, test_results_dir, "last_model"
    )
    print(f"Test Loss: {test_loss:.6f} | PSNR: {test_psnr:.2f} | SSIM: {test_ssim:.4f} | MAE: {test_mae:.6f}")

    # Test evaluation on best model
    best_model_path = os.path.join(args.save_dir_checkpoints, "best_model.pth")
    if os.path.isfile(best_model_path):
        print("\n--- Test Evaluation (Best Model) ---")
        ckpt = torch.load(best_model_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        best_test_loss, best_test_psnr, best_test_ssim, best_test_mae = testify(
            model, test_loaders, test_loader_names, criterion, device, test_results_dir, "best_model"
        )
        print(f"Test Loss: {best_test_loss:.6f} | PSNR: {best_test_psnr:.2f} | SSIM: {best_test_ssim:.4f} | MAE: {best_test_mae:.6f}")
        print(f"(Best model saved at epoch {ckpt['epoch'] + 1})")
    else:
        print("\nNo best_model.pth found, skipping best model test evaluation.")

    logger.close()
    print("Training complete.")


if __name__ == "__main__":
    main()

