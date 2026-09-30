import os
from src import train_unet, train_unetpp

if __name__ == "__main__":
    root = r"E:\Research\benchmark\2D_benchmark\product"

    # train_unetpp(
    #     save_dir = os.path.join(root, "unetpp_deepsupervision"),
    #     device = "cuda",
    #     max_epoch = 200,
    #     batch_size = 32,
    #     learning_rate = 1e-3,
    #     model_kwargs = {
    #         "in_channels": 2*2,
    #         "out_channels": 128,
    #         "deep_supervision": True,
    #     },
    # )

    # train_unetpp(
    #     save_dir = os.path.join(root, "unetpp_baseline"),
    #     device = "cuda",
    #     max_epoch = 200,
    #     batch_size = 32,
    #     learning_rate = 1e-3,
    #     model_kwargs = {
    #         "in_channels": 2*2,
    #         "out_channels": 128,
    #     },
    # )

    # train_unet(
    #     save_dir = os.path.join(root, "unet_gradient"),
    #     device = "cuda",
    #     max_epoch = 200,
    #     batch_size = 32,
    #     learning_rate = 1e-3,
    #     model_kwargs = {
    #         "in_channels": 2*2,
    #         "out_channels": 128,
    #     },
    # )

    # train_unet(
    #     save_dir = os.path.join(root, "unet_baseline"),
    #     device = "cuda",
    #     max_epoch = 200,
    #     batch_size = 32,
    #     learning_rate = 1e-3,
    #     model_kwargs = {
    #         "in_channels": 2,
    #         "out_channels": 128,
    #     },
    # )