## Context

In many modern deep‑learning frameworks, weights and activations are quantised to the FP8 format for memory efficiency.  
A common strategy is *dynamic scaling*: each tensor (or group of tensors) is multiplied by a scalar so that its maximum absolute value fits into the representable range of an 8‑bit signed integer.  

For a real number $x$ and a scale factor $s$, the quantised value is
$$\tilde{x} = \operatorname{round}\!\bigl(\tfrac{x}{s}\bigr),$$
and we require $\max_i |\tilde{x}_i| \le 127$.  
If we choose the scaling divisor to be $448$ (the product of the FP8 exponent range $2^7=128$ and a safety factor $3.5$), then the optimal scale for a tensor $T$ is
$$s_T = \frac{\max_i |T_i|}{448}.$$

In practice we often need two kinds of scales:

* **Per‑tensor scale** – one scalar per weight matrix.
* **Per‑token scale** – one scalar per token (row) in an activation tensor.

The task below asks you to compute these scales for a given weight matrix $W$ and activation tensor $X$.

## Task

Implement the function `fp8_scales(W, X)`:

```python
def fp8_scales(W: np.ndarray, X: np.ndarray) -> Tuple[float, np.ndarray]:
    ...
```

* `W` is a 2‑D NumPy array of shape `(out_dim, in_dim)`.
* `X` is either a 2‑D array of shape `(batch, dim)` or a higher‑dimensional tensor where the last axis contains the feature dimension.  
  In all cases you should treat each *row* (i.e., slice along all axes except the last one) as an independent token.
* The function must return:
  1. A single `float` – the per‑tensor scale for `W`.
  2. A 1‑D NumPy array of shape `(num_tokens,)` – the per‑token scales for `X`.

Both outputs should be computed using the formula above with a divisor of **448**.

## Example

```python
import numpy as np

W = np.array([[0, -3], [4, 1]])
X = np.array([[2, -5], [-1, 7]])

tensor_scale, token_scales = fp8_scales(W, X)

print(tensor_scale)   # 4/448 ≈ 0.008928571428571429
print(token_scales)   # array([5/448, 7/448]) ≈ [0.01116071429, 0.015625]
```

## What the gate checks

The grader computes a reference implementation using NumPy and compares your result with it.  
It reports the global relative L2 error:

$$\mathrm{rel\_err} = \frac{\lVert \hat{s} - s\rVert}{\lVert s\rVert},$$

where $s$ is the concatenation of the tensor scale and all token scales, and $\hat{s}$ is your output.  
The gate requires `rel_err <= 1e-8`.  

A correct implementation will satisfy this bound for a variety of random inputs.
