import os
from src import train_unet

if __name__ == "__main__":
    root = r"E:\Research\benchmark\2D_benchmark"
    
    train_unet(
        save_dir = os.path.join(root, "unet"),
        device = "cuda",
        max_epoch = 200,
        batch_size = 32,
        learning_rate = 1e-3,
        model_kwargs = {
            "in_channels": 2,
            "out_channels": 128,
            "bilinear": False,
        },
    )