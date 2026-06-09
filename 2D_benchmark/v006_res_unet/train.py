import os
import time
import argparse
import torch

from DATASET_PATH import DATASET_PATH
from dataloader import zarr_dataloader
from model_unet import Gravity_Inverse_UNet_2D
from loss import CombinedLoss
from evaluation import psnr, ssim, mae


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
    parser.add_argument("--save_dir", type=str, default="checkpoints")
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


@torch.no_grad()
def validate(model, loaders, criterion, device):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_mae = 0.0
    total_batches = 0

    for loader in loaders:
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

    n = max(total_batches, 1)
    return total_loss / n, total_psnr / n, total_ssim / n, total_mae / n


def save_checkpoint(model, optimizer, scheduler, epoch, best_metric, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_metric": best_metric,
    }, path)


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # Logger (init early to capture full training log)
    writer = None
    if args.logger == "tensorboard":
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(args.tensorboard_log_dir)
    elif args.logger == "swanlab":
        import swanlab
        swanlab.init(
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

    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == "warmup_cosine":
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=args.warmup_epochs)
        main_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, main_sched], milestones=[args.warmup_epochs])
    elif args.scheduler == "warmup_step":
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=args.warmup_epochs)
        main_sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, main_sched], milestones=[args.warmup_epochs])

    # Resume
    start_epoch = 0
    best_metric = -float("inf")
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler and ckpt["scheduler_state_dict"]:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_metric = ckpt["best_metric"]
        print(f"Resumed from epoch {start_epoch}, best_metric={best_metric:.4f}")

    # Training loop
    save_interval = max(1, args.epochs // 5)  # save ~5 checkpoints total
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

        # Logger
        if args.logger == "tensorboard":
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Metrics/psnr", val_psnr, epoch)
            writer.add_scalar("Metrics/ssim", val_ssim, epoch)
            writer.add_scalar("Metrics/mae", val_mae, epoch)
            writer.add_scalar("LR", current_lr, epoch)
        elif args.logger == "swanlab":
            swanlab.log({
                "Loss/train": train_loss,
                "Loss/val": val_loss,
                "Metrics/psnr": val_psnr,
                "Metrics/ssim": val_ssim,
                "Metrics/mae": val_mae,
                "LR": current_lr,
            }, step=epoch)

        # Save best model
        if val_psnr > best_metric:
            best_metric = val_psnr
            save_checkpoint(model, optimizer, scheduler, epoch, best_metric,
                            os.path.join(args.save_dir, "best_model.pth"))
            print(f"  -> New best model saved (psnr={best_metric:.2f})")

        # Periodic checkpoint, only save ~10 models
        if (epoch + 1) % save_interval == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, best_metric,
                            os.path.join(args.save_dir, f"epoch_{epoch+1}.pth"))

    # Final test evaluation
    print("\n--- Test Evaluation ---")
    test_loss, test_psnr, test_ssim, test_mae = validate(model, test_loaders, criterion, device)
    print(f"Test Loss: {test_loss:.6f} | PSNR: {test_psnr:.2f} | SSIM: {test_ssim:.4f} | MAE: {test_mae:.6f}")

    # Save final model
    save_checkpoint(model, optimizer, scheduler, args.epochs - 1, best_metric,
                    os.path.join(args.save_dir, "last_model.pth"))

    if args.logger == "tensorboard":
        writer.close()
    elif args.logger == "swanlab":
        swanlab.finish()
    print("Training complete.")


if __name__ == "__main__":
    main()

