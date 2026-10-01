import os

root = r"E:\Research\gravity_dataset"
dataset_dir = os.path.join(root, "storage", "3D_storage")

DATASET_PATH = {
    "geo_model": {
        "density1": os.path.join(dataset_dir, "geo_model", "density1.zarr"),
        "density2": os.path.join(dataset_dir, "geo_model", "density2.zarr"),
        "density3": os.path.join(dataset_dir, "geo_model", "density3.zarr"),
        "density4": os.path.join(dataset_dir, "geo_model", "density4.zarr"),
        "density5": os.path.join(dataset_dir, "geo_model", "density5.zarr"),
    },
    "fold_layer": {
        "density1": os.path.join(dataset_dir, "fold_layer", "density1.zarr"),
        "density2": os.path.join(dataset_dir, "fold_layer", "density2.zarr"),
        "density3": os.path.join(dataset_dir, "fold_layer", "density3.zarr"),
    }
}

def dataset_path(*args: str):
    value = DATASET_PATH
    for key in args:
        value = value[key]
    return value