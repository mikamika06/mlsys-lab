## Context

Activation-aware weight quantization methods such as AWQ use scaling factors to
move difficult-to-quantize ranges in the weights. A linear layer computes

$$Y = XW^\top,$$

where $X \in \mathbb{R}^{n \times d}$ is an activation matrix and
$W \in \mathbb{R}^{m \times d}$ is a weight matrix.

A scaling vector $s \in \mathbb{R}^{d}$ can be applied to the weights by

$$W' = W \odot s,$$

where each column of $W$ is multiplied by the corresponding scale value. This
operation changes the output unless the activations are compensated:

$$X' = X \oslash s.$$

The corrected computation is

$$X'(W')^\top = (X \oslash s)(W \odot s)^\top = XW^\top.$$

Forgetting the activation compensation is a common implementation bug because
the weight tensor alone appears to be transformed correctly while the layer
output silently changes.

## Task

Implement `restore_awq_equivalence(X, W, s)`.

The function receives three NumPy arrays:

- `X` with shape $(n, d)$ containing activations.
- `W` with shape $(m, d)$ containing weights.
- `s` with shape $(d,)$ containing positive scaling factors.

Return the compensated output matrix after applying the AWQ weight scaling while
preserving the original linear layer output.

The implementation should compute the equivalent of the original output
$XW^\top$ by scaling weights and compensating activations. The returned array
must be a NumPy array with `float64` values.

## Example

```python
import numpy as np

X = np.array([[1.0, 2.0]])
W = np.array([[3.0, 4.0]])
s = np.array([2.0, 5.0])

Y = restore_awq_equivalence(X, W, s)
# Y is [[11.0]]
```

## What the gate checks

The gate compares the implementation against a NumPy oracle that computes the
linear layer after applying the weight scaling and the required inverse scaling
on activations.

The returned matrix must satisfy

$$\max_{i,j}|Y_{ij} - Y^{\mathrm{oracle}}_{ij}| < 10^{-6}.$$

A solution that scales weights but forgets to divide activations by $s$ produces
a numerically different result and fails the gate.
