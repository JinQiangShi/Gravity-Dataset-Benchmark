import os

root = r"E:\Research\gravity_dataset"
dataset_dir = os.path.join(root, "storage", "2D_storage")

DATASET_PATH = {
    "geo_model": {
        "density1": os.path.join(dataset_dir, "geo_model", "density1.zarr"),
        "density2": os.path.join(dataset_dir, "geo_model", "density2.zarr"),
        "density3": os.path.join(dataset_dir, "geo_model", "density3.zarr"),
        "density4": os.path.join(dataset_dir, "geo_model", "density4.zarr"),
        "density5": os.path.join(dataset_dir, "geo_model", "density5.zarr"),
    },
    "salt_model": {
        "density1": os.path.join(dataset_dir, "salt_model", "density1.zarr"),
    }
}

def dataset_path(*args: str):
    value = DATASET_PATH
    for key in args:
        value = value[key]
    return value