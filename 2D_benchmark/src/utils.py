import os
import torch
import random
import swanlab
import numpy as np


def set_seed(seed: int = 42):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多 GPU 情况

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # torch.use_deterministic_algorithms(True)

class Logger:
    def __init__(
        self, 
        log_dir: str,
        project: str, 
        workspace: str, 
    ):
        self._swanlab = swanlab
        swanlab.init(log_dir=log_dir, project=project, workspace=workspace)

    def log_scalars(self, metrics, step):
        """Log a dict of scalar metrics.

        If `metrics` is a list of (value_dict, weight) pairs, the scalar values
        are combined into a single weighted-sum dict before logging.
        """
        if isinstance(metrics, list):
            metrics = self._weighted_sum(metrics)
        self._swanlab.log(metrics, step=step)

    def _weighted_sum(self, metrics: list) -> dict:
        """Weighted-sum a list of (value_dict, weight) pairs into one dict."""
        merged = {}
        for value_dict, weight in metrics:
            for key, value in value_dict.items():
                merged[key] = merged.get(key, 0.0) + weight * value
        return merged

    def close(self):
        self._swanlab.finish()

def extract_file_name(file_path: str):
    """Extract the file name from a file path."""
    return os.path.basename(file_path).split(".")[0]