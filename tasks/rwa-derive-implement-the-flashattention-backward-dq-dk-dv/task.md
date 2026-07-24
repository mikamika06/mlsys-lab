## Context

FlashAttention avoids materializing the full attention matrix during the forward and backward passes. The forward pass computes scaled dot-product attention:

$$S = \frac{QK^\top}{\sqrt{d}},$$

$$P_{ij} = \frac{\exp(S_{ij} - m_i)}{l_i},$$

where $m_i$ and $l_i$ are saved row-wise softmax statistics. The output is

$$O = PV.$$

For a loss with upstream gradient $dO$, the backward pass reconstructs $P$ from $(m,l)$ and computes gradients without storing $P$.

The softmax derivative is

$$dS = P \odot (dP - r),$$

where

$$dP = dO V^\top,$$

and

$$r_i = \sum_j dP_{ij}P_{ij}.$$

The input gradients are

$$dQ = \frac{dS K}{\sqrt{d}},$$

$$dK = \frac{dS^\top Q}{\sqrt{d}},$$

$$dV = P^\top dO.$$

## Task

Implement `flash_backward(Q, K, V, dO, m, l)`:

```python
def flash_backward(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    dO: np.ndarray,
    m: np.ndarray,
    l: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ...
```

The inputs are NumPy arrays. `Q`, `K`, and `V` have shape $(n,d)$ and `dO` has the same shape as the attention output. `m` and `l` contain the saved row-wise softmax maximums and denominators.

Return `(dQ, dK, dV)` as `float64` arrays. Recompute attention probabilities from the saved statistics and implement the closed-form attention backward algorithm without using automatic differentiation.

## Example

```python
import numpy as np

Q = np.array([[1.0, 0.0], [0.0, 1.0]])
K = np.array([[1.0, 0.0], [0.0, 1.0]])
V = np.array([[2.0, 1.0], [0.0, 3.0]])
dO = np.ones((2, 2))

S = Q @ K.T / np.sqrt(2.0)
m = np.max(S, axis=1)
l = np.sum(np.exp(S - m[:, None]), axis=1)

dQ, dK, dV = flash_backward(Q, K, V, dO, m, l)
```

## What the gate checks

The gate constructs a central finite-difference oracle from the naive NumPy attention computation. It compares the returned gradients for $Q$, $K$, and $V$ against this oracle using

$$\max |x_{\mathrm{student}} - x_{\mathrm{oracle}}|.$$

The reported metric must satisfy `max_abs_err <= 1e-4`.
