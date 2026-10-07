import os
from src import visualize

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    root = r"E:\Research\benchmark\3D_benchmark\product"
    visualize(
        npz_dir=os.path.join(root, "unetpp", "test_result"),
        plot_dir=os.path.join(root, "unetpp", "plot")
    )
    
    visualize(
        npz_dir=os.path.join(root, "unet", "test_result"),
        plot_dir=os.path.join(root, "unet", "plot")
    )