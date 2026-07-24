## Context

In transformers with quantized KV caches, Keys $K \in \mathbb{R}^{n \times d}$ and Values $V \in \mathbb{R}^{n \times d}$ are stored at reduced precision. The quantized copies $\hat{K} = K + \epsilon_K$ and $\hat{V} = V + \epsilon_V$ introduce element-wise errors $\epsilon_K, \epsilon_V$.

Standard scaled-dot-product attention computes

$$O = \operatorname{softmax}\!\left(\frac{Q\,K^\top}{\sqrt{d}}\right) V ,$$

where the softmax is applied row-wise over the key dimension. With quantized KV the output becomes

$$\hat{O} = \operatorname{softmax}\!\left(\frac{Q\,\hat{K}^\top}{\sqrt{d}}\right) \hat{V} .$$

The pre-softmax logit perturbation caused by $\epsilon_K$ is $\Delta = Q\,\epsilon_K^\top / \sqrt{d}$. Softmax is a nonlinear function, so $\Delta$ does not map linearly to the output error. The **amplification ratio**

$$\rho = \frac{\operatorname{MSE}(O,\,\hat{O})}{\operatorname{MSE}_{\text{KV}}}$$

quantifies how the softmax nonlinearity amplifies or attenuates the input error, where

$$\operatorname{MSE}_{\text{KV}} = \frac{1}{2}\Bigl[\operatorname{mean}(\epsilon_K^2) + \operatorname{mean}(\epsilon_V^2)\Bigr]$$

and $\operatorname{MSE}(O,\hat{O}) = \operatorname{mean}\!\bigl((O - \hat{O})^2\bigr)$. When $\rho > 1$ the nonlinearity amplifies quantization noise; when $\rho < 1$ it compresses it.

## Task

Implement `kv_quant_error_propagation(Q, K, V, K_hat, V_hat, scale=None)`:

```python
def kv_quant_error_propagation(Q, K, V, K_hat, V_hat, scale=None):
    """
    Compute how KV quantization error propagates through softmax attention.

    Args:
        Q:     (seq_q, d)  query matrix, float64
        K:     (seq_kv, d) original key matrix, float64
        V:     (seq_kv, d) original value matrix, float64
        K_hat: (seq_kv, d) quantized (approximate) key matrix, float64
        V_hat: (seq_kv, d) quantized (approximate) value matrix, float64
        scale: float or None — attention logit scale; defaults to 1/sqrt(d)

    Returns:
        dict with keys:
            output_mse             — mean((O - O_hat)**2)
            kv_error               — mean-KV-dequant MSE
            amplification          — output_mse / kv_error
    """
```

Compute the reference attention output $O$ and the perturbed output $\hat{O}$ using numerically stable softmax (subtract the row-max before exponentiating). Then return the three quantities defined above. Use NumPy only — no PyTorch or TensorFlow.

## Example

```python
import numpy as np
d = 4
scale = 1.0 / np.sqrt(d)
Q = np.eye(4)
K = np.eye(4)
V = np.ones((4, 2))
K_hat = K + 0.05 * np.ones_like(K)
V_hat = V + 0.05 * np.ones_like(V)
result = kv_quant_error_propagation(Q, K, V, K_hat, V_hat, scale=scale)
# result["output_mse"]  — some positive float
# result["kv_error"]    — 0.005
# result["amplification"] — output_mse / kv_error
```

## What the gate checks

The gate returns the mean squared error between the learner's three output values and a NumPy oracle's values, averaged over five test cases that vary $d \in \{8, 16, 32\}$, sequence lengths, and noise levels $\{0.001, 0.01, 0.1\}$. This metric must be $\le 10^{-6}$. The oracle computes the exact same formulas with the same numerically stable softmax. A solution that hardcodes a wrong formula or returns `NotImplementedError` will produce a large MSE and fail.
