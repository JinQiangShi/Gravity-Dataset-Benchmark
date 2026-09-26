# Loss Module Reference

Loss functions under `src/loss`, used for deep-learning-based 2D gravity inversion.
The predicted density model is denoted $\hat{m}$ (`pred` in code), and the ground-truth
density model is denoted $m$ (`target` in code). Both tensors have shape
$(B, 1, n_z, n_x)$, where $B$ is the batch size and $n_z$, $n_x$ are the number of cells
along depth and horizontal directions respectively.
$N = B \cdot n_z \cdot n_x$ is the total number of elements (computed as `pred.numel()` in code).

All losses are `nn.Module`s whose `forward` returns a scalar, and smaller values are better.

---

## 1. MSELoss

Mean squared error, the most basic data-fitting term.

$$
\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} (\hat{m}_i - m_i)^2
$$

- File: `MSELoss.py`
- Parameters: none
- Input: `(pred, target)`

---

## 2. HuberLoss

Huber loss, which reduces to MSE for small errors and to L1 for large errors,
balancing convergence stability and robustness to outliers.

$$
\mathcal{L}_{\delta}(a) =
\begin{cases}
\dfrac{1}{2} a^2, & |a| \le \delta \\[2mm]
\delta \left( |a| - \dfrac{1}{2}\delta \right), & |a| > \delta
\end{cases}
\qquad
\mathcal{L}_{\text{Huber}} = \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}_{\delta}(\hat{m}_i - m_i)
$$

- File: `HuberLoss.py`
- Parameters: `delta` (default `1.0`), the threshold; smaller values behave more like L1
- Input: `(pred, target)`

---

## 3. DepthWeightedHuberLoss

Depth-weighted Huber loss. Built on the Huber loss, it applies a weighting along the
depth direction to counteract the skin effect of potential-field inversion
(over-focusing on shallow cells).

$$
w(z) = (z + z_0)^{-\beta/2}, \qquad
\tilde{w}(z) = \frac{w(z)}{\frac{1}{n_z}\sum_{z'} w(z')}
$$

$$
\mathcal{L}_{\text{DepthHuber}} =
\frac{1}{N} \sum_{i=1}^{N} \tilde{w}(z_i)\, \mathcal{L}_{\delta}(\hat{m}_i - m_i)
$$

where $z$ is the depth index (from 1 to $n_z$), $\tilde{w}$ is the mean-normalized weight,
and $\mathcal{L}_{\delta}$ is the element-wise Huber form above.

- File: `DepthWeightedHuberLoss.py`
- Parameters: `delta` (default `1.0`), `z0` (default `1.0`, stabilization term),
  `beta` (default `2.0`, depth exponent)
- Input: `(pred, target)`

---

## 4. TotalVariationLoss

Anisotropic L1 total variation, a smoothness constraint that suppresses high-frequency
noise and oscillatory solutions. To avoid the non-differentiability of $|\cdot|$ at 0,
it uses the smooth approximation $\sqrt{d^2 + \epsilon}$.

$$
\mathcal{L}_{\text{TV}} =
\frac{1}{N} \left[
\sum_{i,j} \sqrt{( \hat{m}_{i,j+1} - \hat{m}_{i,j} )^2 + \epsilon}
+
\sum_{i,j} \sqrt{( \hat{m}_{i+1,j} - \hat{m}_{i,j} )^2 + \epsilon}
\right]
$$

- File: `TotalVariationLoss.py`
- Parameters: `epsilon` (default `1e-8`, fixed internally)
- Input: `pred` (prediction only)

---

## 5. MinimumSupportLoss

Minimum Support Functional (MSF). It penalizes the region where the model amplitude is
non-zero, driving the inversion toward compact anomalous bodies and shrinking the
support of the solution.

$$
\mathcal{L}_{\text{MS}} =
\frac{1}{N} \sum_{i=1}^{N} \frac{\hat{m}_i^2}{\hat{m}_i^2 + \beta^2}
$$

- File: `MinimumSupportLoss.py`
- Parameters: `beta` (default `1.0`), the threshold scale controlling support determination
- Input: `pred` (prediction only)

---

## 6. MinimumGradientSupportLoss

Minimum Gradient Support Functional (MGS). It shrinks the support of the model gradient,
sharpening the property interfaces and yielding clearer blocky structures.

$$
\mathcal{L}_{\text{MGS}} =
\frac{1}{N} \left[
\sum_{i,j} \frac{(\hat{m}_{i,j+1} - \hat{m}_{i,j})^2}{(\hat{m}_{i,j+1} - \hat{m}_{i,j})^2 + \beta^2}
+
\sum_{i,j} \frac{(\hat{m}_{i+1,j} - \hat{m}_{i,j})^2}{(\hat{m}_{i+1,j} - \hat{m}_{i,j})^2 + \beta^2}
\right]
$$

- File: `MinimumGradientSupportLoss.py`
- Parameters: `beta` (default `1.0`), the threshold scale controlling gradient support determination
- Input: `pred` (prediction only)

---

## 7. SSIMLoss

Structural similarity loss. It measures the structural similarity between the prediction
and the ground truth, being more sensitive to overall morphology (shape, interface
continuity) than pixel-wise errors, and can mitigate stripe-like artifacts.

Local statistics are obtained by convolution with a Gaussian window $w$
(window length `window_size`, standard deviation `sigma`):

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

where $L$ is the data dynamic range (`data_range`), and the overline denotes the mean
over all spatial locations.

- File: `SSIMLoss.py`
- Parameters: `window_size` (default `11`), `sigma` (default `1.5`),
  `data_range` (default `1.0`; set according to the actual data magnitude)
- Input: `(pred, target)`

---

## 8. CombinedLoss

Weighted combined loss, summing the terms above with their weights:

$$
\mathcal{L}_{\text{total}} =
\lambda_{\text{huber}} \mathcal{L}_{\text{Huber}}
+ \lambda_{\text{depth}} \mathcal{L}_{\text{DepthHuber}}
+ \lambda_{\text{tv}} \mathcal{L}_{\text{TV}}
+ \lambda_{\text{ms}} \mathcal{L}_{\text{MS}}
+ \lambda_{\text{mgs}} \mathcal{L}_{\text{MGS}}
+ \lambda_{\text{ssim}} \mathcal{L}_{\text{SSIM}}
$$

- File: `CombinedLoss.py`
- Parameters (defaults):

| Parameter | Default | Corresponding term |
| --- | --- | --- |
| `huber_weight` | `1.0` | Huber data fitting |
| `depth_weight` | `1.0` | Depth-weighted Huber |
| `tv_weight` | `0.1` | Total variation smoothness |
| `ms_weight` | `0.1` | Minimum support |
| `mgs_weight` | `0.1` | Minimum gradient support |
| `ssim_weight` | `0.0` | Structural similarity |

- Input: `(pred, target)`

> Note: setting a weight to `0.0` disables the corresponding term.