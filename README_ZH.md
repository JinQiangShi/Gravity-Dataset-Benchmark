# Gravity Dataset Benchmark

基于 Gravity Dataset 的重力数据密度反演基准测试项目，提供 2D/3D 两种维度的完整训练与评估流程，支持 UNet / UNet++ 网络进行重力异常到密度模型的深度学习反演。

## 项目概览

本项目为地球物理重力反演研究提供标准化基准测试框架，包含以下内容：

- **数据集加载**：从 Zarr 存储中读取密度模型与重力数据，自动划分训练/验证/测试集
- **重力特征增强**：将重力数据插值对齐至密度模型网格，并附加空间梯度特征作为网络输入
- **模型库**：内置 UNet 与 UNet++（支持深监督）两种反演网络
- **组合损失**：集成 Huber、深度加权 Huber、SSIM、全变分、最小支撑、最小梯度支撑等多种损失
- **组合评估**：以 MAE、PSNR、SSIM 加权组合作为模型选择指标
- **实验管理**：基于 SwanLab 记录训练日志与指标曲线

## 目录结构

```
benchmark/
├── 2D_benchmark/                  	# 2D 重力反演基准
│   ├── src/                       	# 核心源码
│   │   ├── dataloader/            	# 数据加载（Zarr 读取与特征构建）
│   │   ├── evaluate/             	# 评估指标（MAE / PSNR / SSIM）
│   │   ├── loss/                 	# 损失函数模块
│   │   ├── model/                	# 网络模型（unet / unetpp）
│   │   ├── scheduler.py          	# Warmup + 余弦退火学习率调度
│   │   ├── train_unet.py         	# UNet 训练流程
│   │   ├── train_unetpp.py       	# UNet++ 训练流程
│   │   ├── utils.py              	# 随机种子设置与 SwanLab 日志封装
│   │   └── visualize.py          	# 反演结果可视化
│   ├── train.py                   	# 2D 训练入口
│   └── visualize.py               	# 2D 可视化入口
├── 3D_benchmark/                  	# 3D 重力反演基准
│   ├── src/                       	# 核心源码（结构与 2D 一致）
│   ├── train.py                   	# 3D 训练入口
│   └── visualize.py               	# 3D 可视化入口
├── environment.yml                	# Conda 环境配置
├── README.md                      	# 英文说明文档
└── README_ZH.md                   	# 中文说明文档
```

## 数据集

数据来自 `gravity_dataset` 项目的统一 Zarr 存储，通过各维度的 `src/dataloader/DATASET_PATH.py` 配置根路径：

| 维度 | 根目录 | 数据集 | 密度网格 |
| :--- | :--- | :--- | :--- |
| 2D | `gravity_dataset/storage/2D_storage` | geo_model | 128 × 128 |
| 3D | `gravity_dataset/storage/3D_storage` | geo_model | 128 × 128 × 128 |

每个 Zarr 文件包含：

- 密度模型标签
- 重力观测数据

数据按 **60% / 20% / 20%** 划分为训练集 / 验证集 / 测试集。

## 数据处理流程

在 [`dataset.py`](2D_benchmark/src/dataloader/dataset.py) 中完成重力数据的特征构建：

1. **插值对齐**：将重力数据插值到密度模型网格尺寸
2. **梯度特征**：沿空间方向计算重力梯度特征
3. **特征拼接**：将原始重力与梯度特征在通道维拼接作为网络输入

## 模型与训练

### 网络

- **UNet**：U 型编解码网络
- **UNet++**：嵌套 U 型结构，支持深监督（deep supervision），多尺度输出按权重 `[0.05, 0.15, 0.3, 0.5]` 加权求和

### 损失函数

[`CombinedLoss`](2D_benchmark/src/loss/CombinedLoss.py) 各分项及默认权重：

| 损失项 | 权重 | 说明 |
| :--- | :--- | :--- |
| Huber | 1.0 | 基础回归损失 |
| DepthWeightedHuber | 1.0 | 深度加权损失 |
| SSIM | 0.03 | 结构相似性损失 |
| TotalVariation | 0.03 | 全变分，抑制噪声 |
| MinimumSupport | 0.03 | 最小支撑约束 |
| MinimumGradientSupport | 0.03 | 最小梯度支撑约束 |

### 评估指标

[`CombinedMetric`](2D_benchmark/src/evaluate/CombinedMetric.py) 将 MAE、PSNR、SSIM 归一化后加权组合（权重 1.0 / 0.5 / 0.3）作为验证集上的模型选择与早停依据：
$$
\text{metric} = w_{\text{mae}} \cdot \text{MAE} + w_{\text{psnr}} \cdot \left(1 - \frac{\text{PSNR}}{\text{psnr\_max}}\right) + w_{\text{ssim}} \cdot (1 - \text{SSIM})
$$

### 训练策略

- 优化器：Adam，学习率 1e-3
- 学习率调度：Warmup（20 epoch）+ 余弦退火
- 梯度裁剪：L2 范数上限 1.0
- 随机种子：固定为 42，保证可复现
- 保存策略：最优模型（`best.pth`）、周期性检查点、最终模型（`final.pth`）
- 早停：验证指标连续多个 epoch 无提升即停止

## 环境配置

见  [`environment.yml`](environment.yml) 

## 使用流程

### 1. 配置数据集路径

编辑对应维度的 [`DATASET_PATH.py`](2D_benchmark/src/dataloader/DATASET_PATH.py)，将 `root` 指向 `gravity_dataset` 项目路径。

### 2. 训练

在 [`train.py`](2D_benchmark/train.py) 中取消注释所需模型配置并运行：

```bash
python 2D_benchmark/train.py
python 3D_benchmark/train.py
```

训练产物保存在指定的 `save_dir` 下：

```
save_dir/
├── log/               # SwanLab 日志
├── checkpoint/        # 模型检查点（best / final / epoch_xxx）
└── test_result/       # 测试集预测结果（.npz）
```

### 3. 可视化

训练完成后单独运行可视化脚本（与训练分离，以避免 OpenMP 库冲突）：

```bash
python 2D_benchmark/visualize.py
python 3D_benchmark/visualize.py
```

从 `test_result` 目录读取 `.npz` 结果，在 `plot` 目录中生成预测对比图。

## 训练结果查看

[SwanLab](https://swanlab.cn/@sjq/Gravity-Dataset/v1/jhumuf/overview)