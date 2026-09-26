import os
from src import main_train

if __name__ == "__main__":
    root = r"E:\Research\benchmark\2D_benchmark"
    
    main_train(
        save_dir = os.path.join(root, "unet"),
        device = "cuda",
        max_epoch = 200,
        batch_size = 32,
        model_type = "unet",
        model_kwargs = {
            "in_channels": 2,
            "out_channels": 128,
            "bilinear": False,
        },
    )