# Gravity Dataset Benchmark

A gravity data density inversion benchmark built on the Gravity Dataset, providing complete training and evaluation pipelines in both 2D and 3D, and supporting UNet / UNet++ networks for deep-learning inversion from gravity anomalies to density models.

## Project Overview

This project provides a standardized benchmark framework for geophysical gravity inversion research, including:

- **Dataset loading**: read density models and gravity data from Zarr storage, with automatic train/val/test splitting
- **Gravity feature enhancement**: interpolate gravity data to align with the density model grid, and append spatial gradient features as network input
- **Model zoo**: built-in UNet and UNet++ (with deep supervision) inversion networks
- **Combined loss**: integrates Huber, depth-weighted Huber, SSIM, total variation, minimum support, minimum gradient support, and more
- **Combined evaluation**: weighted combination of MAE, PSNR, and SSIM as the model selection metric
- **Experiment management**: training logs and metric curves recorded via SwanLab

## Directory Structure

```
benchmark/
├── 2D_benchmark/                  	# 2D gravity inversion benchmark
│   ├── src/                       	# core source code
│   │   ├── dataloader/            	# data loading (Zarr reading and feature construction)
│   │   ├── evaluate/             	# evaluation metrics (MAE / PSNR / SSIM)
│   │   ├── loss/                 	# loss function modules
│   │   ├── model/                	# network models (unet / unetpp)
│   │   ├── scheduler.py          	# Warmup + cosine annealing LR scheduler
│   │   ├── train_unet.py         	# UNet training pipeline
│   │   ├── train_unetpp.py       	# UNet++ training pipeline
│   │   ├── utils.py              	# random seed setup and SwanLab logging wrapper
│   │   └── visualize.py          	# inversion result visualization
│   ├── train.py                   	# 2D training entry point
│   └── visualize.py               	# 2D visualization entry point
├── 3D_benchmark/                  	# 3D gravity inversion benchmark
│   ├── src/                       	# core source code (structure identical to 2D)
│   ├── train.py                   	# 3D training entry point
│   └── visualize.py               	# 3D visualization entry point
├── environment.yml                	# Conda environment configuration
├── README.md                      	# English documentation
└── README_ZH.md                   	# Chinese documentation
```

## Dataset

Data comes from the unified Zarr storage of the `gravity_dataset` project, with the root path configured via `src/dataloader/DATASET_PATH.py` for each dimension:

| Dimension | Root Directory | Dataset | Density Grid |
| :--- | :--- | :--- | :--- |
| 2D | `gravity_dataset/storage/2D_storage` | geo_model | 128 × 128 |
| 3D | `gravity_dataset/storage/3D_storage` | geo_model | 128 × 128 × 128 |

Each Zarr file contains:

- Density model label
- Gravity observation data

Data is split into train / val / test sets at **60% / 20% / 20%**.

## Data Processing Pipeline

Gravity data feature construction is performed in [`dataset.py`](2D_benchmark/src/dataloader/dataset.py):

1. **Interpolation alignment**: interpolate gravity data to the density model grid size
2. **Gradient features**: compute gravity gradient features along spatial directions
3. **Feature concatenation**: concatenate raw gravity and gradient features along the channel dimension as network input

## Model and Training

### Networks

- **UNet**: U-shaped encoder-decoder network
- **UNet++**: nested U-shaped structure with deep supervision; multi-scale outputs are weighted-summed with weights `[0.05, 0.15, 0.3, 0.5]`

### Loss Function

Components and default weights of [`CombinedLoss`](2D_benchmark/src/loss/CombinedLoss.py):

| Loss Term | Weight | Description |
| :--- | :--- | :--- |
| Huber | 1.0 | Base regression loss |
| DepthWeightedHuber | 1.0 | Depth-weighted loss |
| SSIM | 0.03 | Structural similarity loss |
| TotalVariation | 0.03 | Total variation, suppresses noise |
| MinimumSupport | 0.03 | Minimum support constraint |
| MinimumGradientSupport | 0.03 | Minimum gradient support constraint |

### Evaluation Metric

[`CombinedMetric`](2D_benchmark/src/evaluate/CombinedMetric.py) normalizes and weights MAE, PSNR, and SSIM (weights 1.0 / 0.5 / 0.3) as the model selection and early-stopping criterion on the validation set:

$$
\text{metric} = w_{\text{mae}} \cdot \text{MAE} + w_{\text{psnr}} \cdot \left(1 - \frac{\text{PSNR}}{\text{psnr\_max}}\right) + w_{\text{ssim}} \cdot (1 - \text{SSIM})
$$

### Training Strategy

- Optimizer: Adam, learning rate 1e-3
- LR schedule: Warmup (20 epochs) + cosine annealing
- Gradient clipping: L2 norm limit 1.0
- Random seed: fixed at 42 for reproducibility
- Saving strategy: best model (`best.pth`), periodic checkpoints, final model (`final.pth`)
- Early stopping: stops when the validation metric does not improve for multiple consecutive epochs

## Environment Setup

See [`environment.yml`](environment.yml).

## Usage

### 1. Configure Dataset Paths

Edit [`DATASET_PATH.py`](2D_benchmark/src/dataloader/DATASET_PATH.py) for the corresponding dimension and point `root` to the `gravity_dataset` project path.

### 2. Training

Uncomment the desired model configuration in [`train.py`](2D_benchmark/train.py) and run:

```bash
python 2D_benchmark/train.py
python 3D_benchmark/train.py
```

Training artifacts are saved under the specified `save_dir`:

```
save_dir/
├── log/               # SwanLab logs
├── checkpoint/        # model checkpoints (best / final / epoch_xxx)
└── test_result/       # test set prediction results (.npz)
```

### 3. Visualization

After training, run the visualization script separately (kept separate from training to avoid OpenMP library conflicts):

```bash
python 2D_benchmark/visualize.py
python 3D_benchmark/visualize.py
```

It reads `.npz` results from the `test_result` directory and generates prediction comparison plots in the `plot` directory.

## Training Results

[SwanLab](https://swanlab.cn/@sjq/Gravity-Dataset/v1/jhumuf/overview)