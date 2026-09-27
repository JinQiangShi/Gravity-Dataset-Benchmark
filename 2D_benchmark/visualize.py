import os
from src import visualize

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    visualize(
        npz_dir=r"E:\Research\benchmark\2D_benchmark\unet\test_result",
        plot_dir=r"E:\Research\benchmark\2D_benchmark\unet\plot"
    )