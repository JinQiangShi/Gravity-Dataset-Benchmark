import os
from src import train_unet, train_unetpp

if __name__ == "__main__":
    root = r"E:\Research\benchmark\3D_benchmark\product"

    train_unetpp(
        save_dir = os.path.join(root, "unetpp"),
        device = "cuda",
        max_epoch = 100,
        batch_size = 4,
        learning_rate = 1e-3,
        loss_kwargs = {"huber_weight": 1.0, "depth_weight": 1.0, "ssim_weight": 0.03, "tv_weight": 0.03, "ms_weight": 0.03, "mgs_weight": 0.03, "data_range": 1.0},
        metric_kwargs = {"mae_weight": 1.0, "psnr_weight": 0.5, "ssim_weight": 0.3, "psnr_max": 25.0, "data_range": 1.0},
        model_kwargs = {"in_channels": 3*3, "out_channels": 128, "deep_supervision": True},
    )

    # train_unet(
    #     save_dir = os.path.join(root, "unet"),
    #     device = "cuda",
    #     max_epoch = 100,
    #     batch_size = 4,
    #     learning_rate = 1e-3,
    #     loss_kwargs = {"huber_weight": 1.0, "depth_weight": 1.0, "ssim_weight": 0.03, "tv_weight": 0.03, "ms_weight": 0.03, "mgs_weight": 0.03, "data_range": 1.0},
    #     metric_kwargs = {"mae_weight": 1.0, "psnr_weight": 0.5, "ssim_weight": 0.3, "psnr_max": 25.0, "data_range": 1.0},
    #     model_kwargs = {"in_channels": 3*3, "out_channels": 128},
    # )