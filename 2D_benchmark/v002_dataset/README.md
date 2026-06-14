# 版本说明

`v002`版本：基于`v001`版本，对数据集进行消融实验。

# 训练命令

## Z-Score归一化

```PowerShell
# Training command in PowerShell
python -u train.py `
--epochs 500 `
--device cuda `
--dataset_type geo_model `
--batch_size 32 `
--num_workers 8 `
--tv_weight 0.01 `
--lr 1e-4 `
--weight_decay 1e-4 `
--scheduler warmup_plateau `
--plateau_epochs 5 `
--plateau_factor 0.5 `
--warmup_epochs 10 `
--best_metric psnr `
--early_stopping_patience 30 `
--early_stopping_min_delta 1e-4 `
--save_dir_checkpoints checkpoints/geo_model/z-score `
--save_dir_test_results test_results/geo_model/z-score `
--logger swanlab `
--swanlab_project Gravity_Dataset_Benchmark `
--swanlab_workspace sjq `
```

## Min-Max归一化

```
# Training command in PowerShell
python -u train.py `
--epochs 500 `
--device cuda `
--dataset_type geo_model `
--batch_size 32 `
--num_workers 8 `
--tv_weight 0.01 `
--lr 1e-4 `
--weight_decay 1e-4 `
--scheduler warmup_plateau `
--plateau_epochs 5 `
--plateau_factor 0.5 `
--warmup_epochs 10 `
--best_metric psnr `
--early_stopping_patience 30 `
--early_stopping_min_delta 1e-4 `
--save_dir_checkpoints checkpoints/geo_model/min-max `
--save_dir_test_results test_results/geo_model/min-max `
--logger swanlab `
--swanlab_project Gravity_Dataset_Benchmark `
--swanlab_workspace sjq `
```

## Noise（$5\%$高斯噪声）

```
# Training command in PowerShell
python -u train.py `
--epochs 500 `
--device cuda `
--dataset_type geo_model `
--batch_size 32 `
--num_workers 8 `
--tv_weight 0.01 `
--lr 1e-4 `
--weight_decay 1e-4 `
--scheduler warmup_plateau `
--plateau_epochs 5 `
--plateau_factor 0.5 `
--warmup_epochs 10 `
--best_metric psnr `
--early_stopping_patience 30 `
--early_stopping_min_delta 1e-4 `
--save_dir_checkpoints checkpoints/geo_model/noise `
--save_dir_test_results test_results/geo_model/noise `
--logger swanlab `
--swanlab_project Gravity_Dataset_Benchmark `
--swanlab_workspace sjq `
```

## Augment（左右翻转）

```
# Training command in PowerShell
python -u train.py `
--epochs 500 `
--device cuda `
--dataset_type geo_model `
--batch_size 32 `
--num_workers 8 `
--tv_weight 0.01 `
--lr 1e-4 `
--weight_decay 1e-4 `
--scheduler warmup_plateau `
--plateau_epochs 5 `
--plateau_factor 0.5 `
--warmup_epochs 10 `
--best_metric psnr `
--early_stopping_patience 30 `
--early_stopping_min_delta 1e-4 `
--save_dir_checkpoints checkpoints/geo_model/augment `
--save_dir_test_results test_results/geo_model/augment `
--logger swanlab `
--swanlab_project Gravity_Dataset_Benchmark `
--swanlab_workspace sjq `
```

# 推理命令

对现有数据集进行推理

```
python infer.py `
--device cuda `
--checkpoint checkpoints\geo_model\best_model.pth, checkpoints\geo_model\last_model.pth `
--dataset_type geo_model `
--split test `
--batch_size 32 `
--save_dir test_results `
--sub_folder geo_model\best_model, geo_model\last_model `
--save_format zarr `
```

对其它重力数据进行推理

```
python infer.py `
--checkpoint checkpoints\geo_model\best_model.pth, checkpoints\geo_model\last_model.pth `
--input E:\Research\benchmark\2D_benchmark\huoqiu_unet_test\gravity_huoqiu\config1\model\gravity_huoqiu.npz `
--target_nx 126 `
--input_key gravity `
--save_dir E:\Research\benchmark\2D_benchmark\huoqiu_unet_test `
--sub_folder v001\best_model, v001\last_model `
--save_format npz `
```

