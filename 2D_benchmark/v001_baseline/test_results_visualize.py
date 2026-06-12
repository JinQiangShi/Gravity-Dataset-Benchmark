import os
import zarr
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt


def figure_setup(n_z, n_x):
    """
    Set up the figure and axes for plotting.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    dummy_density_true = np.zeros((n_z, n_x))
    density_img_true = axes[0].imshow(dummy_density_true, cmap="viridis", aspect='auto')
    axes[0].set_title("True Density Model", fontsize=12)
    axes[0].set_xlabel("X (Grid Points)")
    axes[0].set_ylabel("Z (Grid Points)")
    plt.colorbar(density_img_true, ax=axes[0], label="True Density Model")

    dummy_density_pred = np.zeros_like(dummy_density_true)
    density_img_pred = axes[1].imshow(dummy_density_pred, cmap="viridis", aspect='auto')
    axes[1].set_title("Pred Density Model", fontsize=12)
    axes[1].set_xlabel("X (Grid Points)")
    axes[1].set_ylabel("Z (Grid Points)")
    plt.colorbar(density_img_pred, ax=axes[1], label="Pred Density Model")
    
    return fig, axes, density_img_true, density_img_pred

def visualize_test_result(label_list:list[list[np.ndarray]], pred_list:list[list[np.ndarray]], save_dir_list:list[str]):
    """
    Visualize the test result.
    """
    n_z, n_x = label_list[0].shape[-2:]
    fig, axes, density_img_true, density_img_pred = figure_setup(n_z, n_x)

    for _, (label, pred, save_dir) in enumerate(zip(label_list, pred_list, save_dir_list)):
        for idx, (_label, _pred) in tqdm(enumerate(zip(label, pred))):
            density_img_true.set_data(_label)
            density_img_true.set_clim(0, 1)

            density_img_pred.set_data(_pred)
            density_img_pred.set_clim(0, 1)

            save_path = os.path.join(save_dir, f"{idx}.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)

    plt.close()
    plt.clf()

def load_zarr_test_results(zarr_file_path:str, max_vis:int):
    """
    Load the test results from Zarr file.
    """
    zarr_file = zarr.open(zarr_file_path)
    
    label_shape = zarr_file["label"].shape
    max_vis = min(max_vis, label_shape[0])
    n_z, n_x = label_shape[-2:]
    
    label = zarr_file["label"][:max_vis]
    label = label.reshape(-1, n_z, n_x)
    
    pred = zarr_file["pred"][:max_vis]
    pred = pred.reshape(-1, n_z, n_x)

    return label, pred

def load_npz_test_results(npz_file_path:str, max_vis:int):
    """
    Load the test results from npz file.
    """
    npz_file = np.load(npz_file_path)
    
    label_shape = npz_file["label"].shape
    max_vis = min(max_vis, label_shape[0])
    n_z, n_x = label_shape[-2:]

    label = npz_file["label"][:max_vis]
    label = label.reshape(-1, n_z, n_x)
    
    pred = npz_file["pred"][:max_vis]
    pred = pred.reshape(-1, n_z, n_x)
    
    return label, pred

def load_test_results(src_dir:str, max_vis:int, dst_dir:str=None):
    """
    Load the test results from the directory.
    """
    label_list = []
    pred_list = []
    save_dir_list = []

    if not dst_dir:
        dst_dir = src_dir

    for file in os.listdir(src_dir):
        if file.endswith(".zarr"):
            label, pred = load_zarr_test_results(os.path.join(src_dir, file), max_vis=max_vis)
            label_list.append(label)
            pred_list.append(pred)
            save_dir_list.append(os.path.join(dst_dir, file.split(".")[0]))
        elif file.endswith(".npz"):
            label, pred = load_npz_test_results(os.path.join(src_dir, file), max_vis=max_vis)
            label_list.append(label)
            pred_list.append(pred)
            save_dir_list.append(os.path.join(dst_dir, file.split(".")[0]))
    return label_list, pred_list, save_dir_list

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    src_dirs = [
        r"E:\Research\benchmark\2D_benchmark\v001_baseline\test_results\geo_model\best_model",
        r"E:\Research\benchmark\2D_benchmark\v001_baseline\test_results\geo_model\last_model"
    ]

    max_vis = 100
    
    for src_dir in src_dirs:
        label_list, pred_list, save_dir_list = load_test_results(src_dir, max_vis=max_vis)
        visualize_test_result(label_list, pred_list, save_dir_list)
