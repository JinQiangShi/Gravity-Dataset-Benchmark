# 版本说明

2D数据集Benchmark的Baseline。

# 训练细节

模型：UNet

数据集：2D Geo Model

损失函数：``MSE`` + ``TVLoss``

优化器：``AdamW``

# 启动命令

```PowerShell
# Training command in PowerShell
python train.py `
--epochs 100 `
--batch_size 32 `
--lr 1e-4 `
--weight_decay 1e-4 `
--num_workers 8 `
--tv_weight 1e-2 `
--scheduler cosine `
--save_dir checkpoints/geo_model `
--dataset_type geo_model `
--device cuda `
--logger swanlab `
--swanlab_project Gravity_Dataset_Benchmark_2D `
--swanlab_workspace sjq
```

```PowerShell
# Training command in PowerShell
Start-Process python -ArgumentList `
    "-u", `
    "train.py", `
    "--epochs", "100", `
    "--batch_size", "32", `
    "--lr", "1e-4", `
    "--weight_decay", "1e-4", `
    "--num_workers", "8", `
    "--tv_weight", "1e-2", `
    "--scheduler", "cosine", `
    "--save_dir", "checkpoints/geo_model", `
    "--dataset_type", "geo_model", `
    "--device", "cuda", `
    "--logger", "swanlab", `
    "--swanlab_project", "Gravity_Dataset_Benchmark", `
    "--swanlab_workspace", "sjq" `
    -RedirectStandardOutput "logs/train.log" `
    -RedirectStandardError "logs/train_err.log" `
    -NoNewWindow -Wait
```

