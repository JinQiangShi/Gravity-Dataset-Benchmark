import os

root_path = r"E:\Research\gravity_dataset\2D_storage"

DATASET_PATH = {
    "geo_model": {
        "density1.zarr": os.path.join(root_path, "geo_model", "density1.zarr"),
        "density2.zarr": os.path.join(root_path, "geo_model", "density2.zarr"),
        "density3.zarr": os.path.join(root_path, "geo_model", "density3.zarr"),
        "density4.zarr": os.path.join(root_path, "geo_model", "density4.zarr"),
        "density5.zarr": os.path.join(root_path, "geo_model", "density5.zarr"),
    },
    "salt_model": {
        "density1.zarr": os.path.join(root_path, "salt_model", "density1.zarr"),
    }
}
