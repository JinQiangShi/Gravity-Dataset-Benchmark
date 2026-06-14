# 版本说明

`v001`版本：Baseline。

# 训练命令

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
--save_dir_checkpoints checkpoints/geo_model `
--save_dir_test_results test_results/geo_model `
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

