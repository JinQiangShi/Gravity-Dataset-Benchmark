# Loss 模块说明

本项目 `src/loss` 下的损失函数，用于基于深度学习的二维重力反演。
记预测密度模型为 $\hat{m}$（代码中的 `pred`），真值密度模型为 $m$（代码中的 `target`），
张量形状均为 $(B, 1, n_z, n_x)$，其中 $B$ 为 batch，$n_z$、$n_x$ 分别为深度与水平方向网格数。
$N = B \cdot n_z \cdot n_x$ 为元素总数（代码中以 `pred.numel()` 计算）。

所有损失均为 `nn.Module`，`forward` 返回标量，且数值越小越好。

---

## 1. MSELoss

均方误差，最基础的数据拟合项。

$$
\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} (\hat{m}_i - m_i)^2
$$

- 文件：`MSELoss.py`
- 参数：无
- 输入：`(pred, target)`

---

## 2. HuberLoss

Huber 损失，小误差时退化为 MSE、大误差时退化为 L1，兼顾收敛稳定性与抗离群值能力。

$$
\mathcal{L}_{\delta}(a) =
\begin{cases}
\dfrac{1}{2} a^2, & |a| \le \delta \\[2mm]
\delta \left( |a| - \dfrac{1}{2}\delta \right), & |a| > \delta
\end{cases}
\qquad
\mathcal{L}_{\text{Huber}} = \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}_{\delta}(\hat{m}_i - m_i)
$$

- 文件：`HuberLoss.py`
- 参数：`delta`（默认 `1.0`），阈值，越小越偏向 L1
- 输入：`(pred, target)`

---

## 3. DepthWeightedHuberLoss

深度加权 Huber 损失。以 Huber 损失为基础，沿深度方向施加权重，
用于压制位场反演的趋肤效应（浅部过度聚焦）。

$$
w(z) = (z + z_0)^{-\beta/2}, \qquad
\tilde{w}(z) = \frac{w(z)}{\frac{1}{n_z}\sum_{z'} w(z')}
$$

$$
\mathcal{L}_{\text{DepthHuber}} =
\frac{1}{N} \sum_{i=1}^{N} \tilde{w}(z_i)\, \mathcal{L}_{\delta}(\hat{m}_i - m_i)
$$

其中 $z$ 为深度索引（从 1 到 $n_z$），$\tilde{w}$ 为均值归一化后的权重，
$\mathcal{L}_{\delta}$ 为上文 Huber 的逐元素形式。

- 文件：`DepthWeightedHuberLoss.py`
- 参数：`delta`（默认 `1.0`）、`z0`（默认 `1.0`，稳定项）、`beta`（默认 `2.0`，深度指数）
- 输入：`(pred, target)`

---

## 4. TotalVariationLoss

各向异性 L1 总变差，平滑约束，抑制高频噪声与震荡解。
为避免 $|\cdot|$ 在 0 处不可导，用 $\sqrt{d^2 + \epsilon}$ 做平滑近似。

$$
\mathcal{L}_{\text{TV}} =
\frac{1}{N} \left[
\sum_{i,j} \sqrt{( \hat{m}_{i,j+1} - \hat{m}_{i,j} )^2 + \epsilon}
+
\sum_{i,j} \sqrt{( \hat{m}_{i+1,j} - \hat{m}_{i,j} )^2 + \epsilon}
\right]
$$

- 文件：`TotalVariationLoss.py`
- 参数：`epsilon`（默认 `1e-8`，内部固定）
- 输入：`pred`（只需预测值）

---

## 5. MinimumSupportLoss

最小体积支撑泛函（Minimum Support Functional, MSF），
惩罚模型幅值非零区域，使反演结果趋于紧凑的异常体、收缩解的支撑。

$$
\mathcal{L}_{\text{MS}} =
\frac{1}{N} \sum_{i=1}^{N} \frac{\hat{m}_i^2}{\hat{m}_i^2 + \beta^2}
$$

- 文件：`MinimumSupportLoss.py`
- 参数：`beta`（默认 `1.0`），控制支撑判定的阈值尺度
- 输入：`pred`（只需预测值）

---

## 6. MinimumGradientSupportLoss

最小梯度支撑泛函（Minimum Gradient Support Functional, MGS），
收缩模型梯度的支撑，锐化物性界面、得到更清晰的块状结构。

$$
\mathcal{L}_{\text{MGS}} =
\frac{1}{N} \left[
\sum_{i,j} \frac{(\hat{m}_{i,j+1} - \hat{m}_{i,j})^2}{(\hat{m}_{i,j+1} - \hat{m}_{i,j})^2 + \beta^2}
+
\sum_{i,j} \frac{(\hat{m}_{i+1,j} - \hat{m}_{i,j})^2}{(\hat{m}_{i+1,j} - \hat{m}_{i,j})^2 + \beta^2}
\right]
$$

- 文件：`MinimumGradientSupportLoss.py`
- 参数：`beta`（默认 `1.0`），控制梯度支撑判定的阈值尺度
- 输入：`pred`（只需预测值）

---

## 7. SSIMLoss

结构相似性损失，衡量预测与真值的结构相似度，
对整体形态（形状、界面连续性）比逐像素误差更敏感，可缓解条纹状伪影。

局部统计量由高斯窗 $w$ 卷积得到（窗长 `window_size`、标准差 `sigma`）：

$$
\mu_{\hat{m}} = w * \hat{m}, \quad \mu_{m} = w * m
$$
$$
\sigma_{\hat{m}}^2 = w * \hat{m}^2 - \mu_{\hat{m}}^2, \quad
\sigma_{m}^2 = w * m^2 - \mu_{m}^2, \quad
\sigma_{\hat{m}m} = w * (\hat{m} m) - \mu_{\hat{m}} \mu_{m}
$$

$$
\text{SSIM} =
\frac{(2\mu_{\hat{m}}\mu_{m} + C_1)(2\sigma_{\hat{m}m} + C_2)}
{(\mu_{\hat{m}}^2 + \mu_{m}^2 + C_1)(\sigma_{\hat{m}}^2 + \sigma_{m}^2 + C_2)}
$$

$$
C_1 = (0.01\,L)^2, \qquad C_2 = (0.03\,L)^2
$$

$$
\mathcal{L}_{\text{SSIM}} = 1 - \overline{\text{SSIM}}
$$

其中 $L$ 为数据动态范围（`data_range`），上划线表示对所有空间位置的均值。

- 文件：`SSIMLoss.py`
- 参数：`window_size`（默认 `11`）、`sigma`（默认 `1.5`）、`data_range`（默认 `1.0`，需按实际数据量级设置）
- 输入：`(pred, target)`

---

## 8. CombinedLoss

加权组合损失，将上述各项按权重求和：

$$
\mathcal{L}_{\text{total}} =
\lambda_{\text{huber}} \mathcal{L}_{\text{Huber}}
+ \lambda_{\text{depth}} \mathcal{L}_{\text{DepthHuber}}
+ \lambda_{\text{tv}} \mathcal{L}_{\text{TV}}
+ \lambda_{\text{ms}} \mathcal{L}_{\text{MS}}
+ \lambda_{\text{mgs}} \mathcal{L}_{\text{MGS}}
+ \lambda_{\text{ssim}} \mathcal{L}_{\text{SSIM}}
$$

- 文件：`CombinedLoss.py`
- 参数（默认值）：

| 参数 | 默认值 | 对应项 |
| --- | --- | --- |
| `huber_weight` | `1.0` | Huber 数据拟合 |
| `depth_weight` | `1.0` | 深度加权 Huber |
| `tv_weight` | `0.1` | 总变差平滑 |
| `ms_weight` | `0.1` | 最小体积支撑 |
| `mgs_weight` | `0.1` | 最小梯度支撑 |
| `ssim_weight` | `0.0` | 结构相似性 |

- 输入：`(pred, target)`

> 注：设置权重为 `0.0` 可关闭对应项。