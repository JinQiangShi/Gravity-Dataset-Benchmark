import os
from src import visualize

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    root = r"E:\Research\benchmark\3D_benchmark\product"

    # ========== Visualize fold_layer ==========
    visualize(
        npz_dir=os.path.join(root, "fold_layer", "unetpp", "test_result"),
        plot_dir=os.path.join(root, "fold_layer", "unetpp", "plot")
    )

    visualize(
        npz_dir=os.path.join(root, "fold_layer", "unet", "test_result"),
        plot_dir=os.path.join(root, "fold_layer", "unet", "plot")
    )
    
    # ========== Visualize geo_model ==========

    # visualize(
    #     npz_dir=os.path.join(root, "geo_model", "unetpp", "test_result"),
    #     plot_dir=os.path.join(root, "geo_model", "unetpp", "plot")
    # )
    
    # visualize(
    #     npz_dir=os.path.join(root, "geo_model", "unet", "test_result"),
    #     plot_dir=os.path.join(root, "geo_model", "unet", "plot")
    # )