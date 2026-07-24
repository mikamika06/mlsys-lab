## Context

In quantization‑aware training a *migration scale* is used to balance the dynamic ranges of activations and weights. For each output channel \(j\) we compute

$$s_j \;=\;\frac{(\max |X_j|)^{\,\alpha}}{(\max |W_j|)^{\,1-\alpha}},$$

where \(X_j\) denotes all activation values that flow through channel \(j\), \(W_j\) the corresponding weight tensor, and \(\alpha\in[0,1]\) is a hyper‑parameter. The numerator pulls the scale up when activations are large; the denominator pushes it down when weights are large.

The task is to implement this computation efficiently for arbitrary tensors that follow the usual PyTorch/NumPy layout:
- `W` has shape \((C_{\text{out}},\,\dots)\) – the first dimension indexes output channels.
- `X` has shape \((N,\,C_{\text{out}},\,\dots)\) – a batch of activations.

The function must return a one‑dimensional NumPy array of type `float64` containing \(s_j\) for every channel.

## Task

Implement the following function:

```python
def compute_migration_scales(W: np.ndarray, X: np.ndarray, alpha: float) -> np.ndarray:
    ...
```

It should compute the per‑channel migration scales defined above. No explicit Python loops are allowed; use vectorised NumPy operations only.

## Example

```python
import numpy as np

W = np.array([[[1, -2], [3, 0]], [[-1, 4], [0, -5]]])   # shape (2,2,2)
X = np.array([
    [[[0.5, 1.0], [-1.5, 2.0]],
     [[-0.5, 0.0], [3.0, -4.0]]],
    [[[1.0, -1.0], [2.0, -2.0]],
     [[0.0, 0.5], [-1.0, 1.5]]]
])  # shape (2,2,2,2)
alpha = 0.3

s = compute_migration_scales(W, X, alpha)
print(s)   # e.g. [0.70710678, 1.41421356]
```

(The numbers above are illustrative; the exact values depend on the data.)

## What the gate checks

The grader computes a reference implementation using NumPy and compares your output with it via the scorer `max_abs_err`. The solution must satisfy

$$\max_j |\,s^{\text{your}}_j - s^{\text{ref}}_j| \;\leq\; 10^{-6}.$$

The function should also return a NumPy array of type `float64` and shape `(C_{\text{out}},)`.
