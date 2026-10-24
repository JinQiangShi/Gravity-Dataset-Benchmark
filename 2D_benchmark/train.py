import os
from src import train_unet, train_unetpp

if __name__ == "__main__":
    root = r"E:\Research\benchmark\2D_benchmark\product"

    # ========== Train on salt_model ==========

    train_unetpp(
        save_dir = os.path.join(root, "salt_model", "unetpp"),
        device = "cuda",
        dataset_type = "salt_model",
        max_epoch = 200,
        batch_size = 32,
        learning_rate = 1e-3,
        add_gradient = True,
        loss_kwargs = {"huber_weight": 1.0, "depth_weight": 1.0, "ssim_weight": 0.03, "tv_weight": 0.01, "ms_weight": 0.01, "mgs_weight": 0.01, "data_range": 1.0},
        metric_kwargs = {"mae_weight": 1.0, "psnr_weight": 0.5, "ssim_weight": 0.3, "psnr_max": 25.0, "data_range": 1.0},
        model_kwargs = {"in_channels": 2*2, "out_channels": 128, "deep_supervision": True},
    )

    # train_unet(
    #     save_dir = os.path.join(root, "salt_model", "unet"),
    #     device = "cuda",
    #     dataset_type = "salt_model",
    #     max_epoch = 200,
    #     batch_size = 32,
    #     learning_rate = 1e-3,
    #     add_gradient = True,
    #     loss_kwargs = {"huber_weight": 1.0, "depth_weight": 1.0, "ssim_weight": 0.03, "tv_weight": 0.01, "ms_weight": 0.01, "mgs_weight": 0.01, "data_range": 1.0},
    #     metric_kwargs = {"mae_weight": 1.0, "psnr_weight": 0.5, "ssim_weight": 0.3, "psnr_max": 25.0, "data_range": 1.0},
    #     model_kwargs = {"in_channels": 2*2, "out_channels": 128},
    # )

    # ========== Train on geo_model ==========

    # train_unetpp(
    #     save_dir = os.path.join(root, "geo_model", "unetpp_deepsupervision"),
    #     device = "cuda",
    #     max_epoch = 200,
    #     batch_size = 32,
    #     learning_rate = 1e-3,
    #     add_gradient = True,
    #     loss_kwargs = {"huber_weight": 1.0, "depth_weight": 1.0, "ssim_weight": 0.03, "tv_weight": 0.03, "ms_weight": 0.03, "mgs_weight": 0.03, "data_range": 1.0},
    #     metric_kwargs = {"mae_weight": 1.0, "psnr_weight": 0.5, "ssim_weight": 0.3, "psnr_max": 25.0, "data_range": 1.0},
    #     model_kwargs = {"in_channels": 2*2, "out_channels": 128, "deep_supervision": True},
    # )

    # train_unetpp(
    #     save_dir = os.path.join(root, "geo_model", "unetpp_baseline"),
    #     device = "cuda",
    #     max_epoch = 200,
    #     batch_size = 32,
    #     learning_rate = 1e-3,
    #     add_gradient = True,
    #     loss_kwargs = {"huber_weight": 1.0, "depth_weight": 1.0, "ssim_weight": 0.03, "tv_weight": 0.03, "ms_weight": 0.03, "mgs_weight": 0.03, "data_range": 1.0},
    #     metric_kwargs = {"mae_weight": 1.0, "psnr_weight": 0.5, "ssim_weight": 0.3, "psnr_max": 25.0, "data_range": 1.0},
    #     model_kwargs = {"in_channels": 2*2, "out_channels": 128},
    # )

    # train_unet(
    #     save_dir = os.path.join(root, "geo_model", "unet_gradient"),
    #     device = "cuda",
    #     max_epoch = 200,
    #     batch_size = 32,
    #     learning_rate = 1e-3,
    #     add_gradient = True,
    #     loss_kwargs = {"huber_weight": 1.0, "depth_weight": 1.0, "ssim_weight": 0.03, "tv_weight": 0.03, "ms_weight": 0.03, "mgs_weight": 0.03, "data_range": 1.0},
    #     metric_kwargs = {"mae_weight": 1.0, "psnr_weight": 0.5, "ssim_weight": 0.3, "psnr_max": 25.0, "data_range": 1.0},
    #     model_kwargs = {"in_channels": 2*2, "out_channels": 128},
    # )

    # train_unet(
    #     save_dir = os.path.join(root, "geo_model", "unet_baseline"),
    #     device = "cuda",
    #     max_epoch = 200,
    #     batch_size = 32,
    #     learning_rate = 1e-3,
    #     add_gradient = False,
    #     loss_kwargs = {"huber_weight": 1.0, "depth_weight": 1.0, "ssim_weight": 0.03, "tv_weight": 0.03, "ms_weight": 0.03, "mgs_weight": 0.03, "data_range": 1.0},
    #     metric_kwargs = {"mae_weight": 1.0, "psnr_weight": 0.5, "ssim_weight": 0.3, "psnr_max": 25.0, "data_range": 1.0},
    #     model_kwargs = {"in_channels": 2, "out_channels": 128},
    # )
