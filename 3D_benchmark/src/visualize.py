"""
Here use hard coded values for the figure size and axis labels.
Please adjust them according to your actual data.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from skimage.measure import marching_cubes
from multiprocessing import Pool, cpu_count
from .utils import extract_file_name


_GEOMETRY = None


def _worker_init():
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMBA_NUM_THREADS"] = "1"

    # 每个子进程只构建一次固定几何元素，之后由所有任务共享
    global _GEOMETRY
    _GEOMETRY = _visualize_grid()


def _visualize_grid():
    density_grid_nx = 128
    density_grid_ny = 128
    density_grid_nz = 128

    data_x_min = -5.0
    data_x_max = 10.0

    data_y_min = -5.0
    data_y_max = 10.0

    _measurement_X = np.linspace(
        data_x_min,
        data_x_max,
        128,
        endpoint=True,
    )
    _measurement_Y = np.linspace(
        data_y_min,
        data_y_max,
        128,
        endpoint=True,
    )

    measurement_X, measurement_Y = np.meshgrid(_measurement_X, _measurement_Y)

    return (
        density_grid_nx,
        density_grid_ny,
        density_grid_nz,
        measurement_X,
        measurement_Y,
    )

def _single_plot(args):
    """
    Worker for a single sample: visualize density and gravity data.
    """
    gravity, density_true, density_pred, output_path = args

    (
        density_grid_nx,
        density_grid_ny,
        density_grid_nz,
        measurement_X,
        measurement_Y,
    ) = _GEOMETRY

    fig = plt.figure(figsize=(21, 6), layout="constrained")

    # plot gravity components
    g_x, g_y, g_z = gravity[:3]
    axes0 = fig.add_subplot(1, 3, 1, projection="3d")
    axes0.plot_surface(measurement_X, measurement_Y, g_x, color="C0", alpha=0.6)
    axes0.plot_surface(measurement_X, measurement_Y, g_y, color="C1", alpha=0.6)
    axes0.plot_surface(measurement_X, measurement_Y, g_z, color="C2", alpha=0.6)
    axes0.set_box_aspect([1, 1, 1])
    axes0.set_title("Gravity Components Comparison", fontsize=12, pad=20)
    axes0.set_xlabel('X Position (km)')
    axes0.set_ylabel('Y Position (km)')
    axes0.set_zlabel('Gravity (mGal)')
    axes0.legend(handles=[
        Patch(facecolor="C0", alpha=0.6, label="Gx"),
        Patch(facecolor="C1", alpha=0.6, label="Gy"),
        Patch(facecolor="C2", alpha=0.6, label="Gz"),
    ], loc='lower center', ncol=3)
    axes0.view_init(elev=5., azim=45)

    # plot true density model
    axes1 = fig.add_subplot(1, 3, 2, projection="3d")
    density_true = density_true.reshape(density_grid_nz, density_grid_ny, density_grid_nx)
    density_true = density_true.transpose(2, 1, 0)
    verts, faces, mormals, values = marching_cubes(density_true, level=0.5)
    axes1.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces, edgecolor='none')
    axes1.set_xlim(0, density_grid_nx)
    axes1.set_ylim(0, density_grid_ny)
    axes1.set_zlim(0, density_grid_nz)
    axes1.invert_zaxis()
    axes1.set_box_aspect([1, 1, 1])
    axes1.set_title("3D Density Model", fontsize=12, pad=20)
    axes1.set_xlabel("X (Grid Points)")
    axes1.set_ylabel("Y (Grid Points)")
    axes1.set_zlabel("Z (Grid Points)")
    axes1.view_init(elev=5, azim=45)

    # plot pred density model
    axes2 = fig.add_subplot(1, 3, 3, projection="3d")
    density_pred = density_pred.reshape(density_grid_nz, density_grid_ny, density_grid_nx)
    density_pred = density_pred.transpose(2, 1, 0)
    density_pred = density_pred > 0.5 # binary
    verts, faces, mormals, values = marching_cubes(density_pred, level=0.5)
    axes2.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces, edgecolor='none')
    axes2.set_xlim(0, density_grid_nx)
    axes2.set_ylim(0, density_grid_ny)
    axes2.set_zlim(0, density_grid_nz)
    axes2.invert_zaxis()
    axes2.set_box_aspect([1, 1, 1])
    axes2.set_title("3D Density Model", fontsize=12, pad=20)
    axes2.set_xlabel("X (Grid Points)")
    axes2.set_ylabel("Y (Grid Points)")
    axes2.set_zlabel("Z (Grid Points)")
    axes2.view_init(elev=5, azim=45)

    plt.savefig(output_path)
    fig.clf()
    plt.close(fig)


def _visualize(
    npz_path: str,
    plot_dir: str,
    n_processes: int = None,
    chunk_size: int = 1,
    max_vis: int = None,
):
    """
    Visualize the density and gravity data from a npz file in parallel.
    """
    npz_data = np.load(npz_path)
    gravity_all = npz_data["gravity"]
    density_true_all = npz_data["density"]
    density_pred_all = npz_data["density_pred"]
    length = len(gravity_all)

    if max_vis is not None:
        max_vis = min(max_vis, length)
    else:
        max_vis = length

    if n_processes is None:
        n_processes = int(cpu_count() / 2)

    def args_generator():
        for i, (gravity, density_true, density_pred) in enumerate(
            zip(gravity_all[:max_vis], density_true_all[:max_vis], density_pred_all[:max_vis])
        ):
            save_path = os.path.join(plot_dir, f"model{i}.png")
            yield gravity.copy(), density_true.copy(), density_pred.copy(), save_path

    with Pool(processes=n_processes, initializer=_worker_init) as pool:
        with tqdm(total=max_vis, desc=f"Visualizing {extract_file_name(npz_path)}") as pbar:
            task_iterator = pool.imap_unordered(
                _single_plot,
                args_generator(),
                chunksize=chunk_size,
            )
            for _ in task_iterator:
                pbar.update(chunk_size)


def visualize(
    npz_dir: str,
    plot_dir: str,
    n_processes: int = None,
    chunk_size: int = 1,
    max_vis: int = None,
):
    """
    Visualize the density and gravity data from a npz directory.
    """
    for npz_file in os.listdir(npz_dir):
        if npz_file.endswith(".npz"):
            npz_file_path = os.path.join(npz_dir, npz_file)
            sub_plot_dir = os.path.join(plot_dir, extract_file_name(npz_file_path))
            os.makedirs(sub_plot_dir, exist_ok=True)
            _visualize(
                npz_file_path,
                sub_plot_dir,
                n_processes=n_processes,
                chunk_size=chunk_size,
                max_vis=max_vis,
            )