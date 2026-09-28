import os
from src import visualize

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    root = r"E:\Research\benchmark\2D_benchmark\product"
    visualize(
        npz_dir=os.path.join(root, "unet_baseline", "test_result"),
        plot_dir=os.path.join(root, "unet_baseline", "plot")
    )