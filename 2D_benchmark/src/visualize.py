"""
Here use hard coded values for the figure size and axis labels.
Please adjust them according to your actual data.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from .utils import extract_file_name

def figure_setup():
    fig, axes = plt.subplots(1, 3, figsize=(21, 5))
    measurement_X = np.linspace(0, 5, 128)
    dummy_gravity = np.zeros_like(measurement_X)
    gx_line = axes[0].plot(measurement_X, dummy_gravity, 'b-', linewidth=1.5, label="Gx (Horizontal)")[0]
    gz_line = axes[0].plot(measurement_X, dummy_gravity, 'r-', linewidth=1.5, label="Gz (Vertical)")[0]
    axes[0].set_title('Gravity Components Comparison', fontsize=12)
    axes[0].set_xlabel('Position (km)')
    axes[0].set_ylabel('Gravity (mGal)')
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].legend()

    dummy_denisty = np.zeros((128, 128))
    density_true_img = axes[1].imshow(dummy_denisty, cmap="viridis")
    axes[1].set_title('Density True', fontsize=12)
    axes[1].set_xlabel('X (Grid Points)')
    axes[1].set_ylabel('Z (Grid Points)')
    plt.colorbar(density_true_img, ax=axes[1], label='Density (kg/m^3)')

    density_pred_img = axes[2].imshow(dummy_denisty, cmap="viridis")
    axes[2].set_title('Density Pred', fontsize=12)
    axes[2].set_xlabel('X (Grid Points)')
    axes[2].set_ylabel('Z (Grid Points)')
    plt.colorbar(density_pred_img, ax=axes[2], label='Density (kg/m^3)')

    return fig, axes, gx_line, gz_line, density_true_img, density_pred_img

def _visualize(
    npz_path:str,
    plot_dir:str
):
    """
    Visualize the density and gravity data from a npz file.
    """
    npz_data = np.load(npz_path)
    gravity = npz_data["gravity"]
    density_true = npz_data["density"]
    density_pred = npz_data["density_pred"]
    length = len(gravity)

    plt.ioff()
    fig, axes, gx_line, gz_line, density_true_img, density_pred_img = figure_setup()

    def single_plot(gravity, density_true, density_pred, output_path):
        """
        Plot gravity data for a single density model.
        """
        gx, gz = gravity[:2]
        gx_line.set_ydata(gx)
        gz_line.set_ydata(gz)
        axes[0].relim()
        axes[0].autoscale_view()

        nz, nx = density_true.shape[-2:]
        density_true_img.set_data(density_true.reshape(nz, nx))
        density_true_img.set_clim(vmin=0, vmax=1)

        density_pred_img.set_data(density_pred.reshape(nz, nx))
        density_pred_img.set_clim(vmin=0, vmax=1)

        fig.savefig(output_path)

    for i, (gravity, density_true, density_pred) in tqdm(
        enumerate(
            zip(gravity, density_true, density_pred)), 
        desc=f"Visualizing {extract_file_name(npz_path)}", total=length):
        save_path = os.path.join(plot_dir, f"model{i}.png")
        single_plot(gravity, density_true, density_pred, save_path)

def visualize(
    npz_dir:str,
    plot_dir:str
):
    """
    Visualize the density and gravity data from a npz directory.
    """
    for npz_file in os.listdir(npz_dir):
        if npz_file.endswith(".npz"):
            npz_file_path = os.path.join(npz_dir, npz_file)
            sub_plot_dir = os.path.join(plot_dir, extract_file_name(npz_file_path))
            os.makedirs(sub_plot_dir, exist_ok=True)
            _visualize(npz_file_path, sub_plot_dir)