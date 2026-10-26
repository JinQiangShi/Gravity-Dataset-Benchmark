import os
from src import visualize

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    root = r"E:\Research\benchmark\2D_benchmark\product"

    # ========== Visualize salt_model ==========
    visualize(
        npz_dir=os.path.join(root, "salt_model", "unetpp", "test_result"),
        plot_dir=os.path.join(root, "salt_model", "unetpp", "plot")
    )

    visualize(
        npz_dir=os.path.join(root, "salt_model", "unet", "test_result"),
        plot_dir=os.path.join(root, "salt_model", "unet", "plot")
    )

    # ========== Visualize geo_model ==========

    # visualize(
    #     npz_dir=os.path.join(root, "geo_model", "unetpp_deepsupervision", "test_result"),
    #     plot_dir=os.path.join(root, "geo_model", "unetpp_deepsupervision", "plot")
    # )

    # visualize(
    #     npz_dir=os.path.join(root, "geo_model", "unetpp_baseline", "test_result"),
    #     plot_dir=os.path.join(root, "geo_model", "unetpp_baseline", "plot")
    # )

    # visualize(
    #     npz_dir=os.path.join(root, "geo_model", "unet_gradient", "test_result"),
    #     plot_dir=os.path.join(root, "geo_model", "unet_gradient", "plot")
    # )
    
    # visualize(
    #     npz_dir=os.path.join(root, "geo_model", "unet_baseline", "test_result"),
    #     plot_dir=os.path.join(root, "geo_model", "unet_baseline", "plot")
    # )