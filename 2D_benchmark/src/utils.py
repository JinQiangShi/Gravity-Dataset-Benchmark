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
        """Log a dict of scalar metrics."""
        self._swanlab.log(metrics, step=step)

    def close(self):
        self._swanlab.finish()

def extract_file_name(file_path: str):
    """Extract the file name from a file path."""
    return os.path.basename(file_path).split(".")[0]