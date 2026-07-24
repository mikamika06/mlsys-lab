## Context

SparseGPT-style pruning reduces the number of stored weights while trying to preserve
the layer output. A common local approximation uses the inverse of an input Hessian
matrix. For a linear layer with weights $W \in \mathbb{R}^{m \times n}$ and calibration
activations $X \in \mathbb{R}^{s \times n}$, the approximate Hessian is

$$
H = \frac{X^\top X}{s} + \lambda I .
$$

The inverse Hessian diagonal estimates the sensitivity of removing each weight. For a
weight $w_{ij}$, a simple saliency score is

$$
S_{ij} = \frac{w_{ij}^2}{(H^{-1})_{jj}} .
$$

For 2:4 structured sparsity, every consecutive group of four input channels must keep
exactly two weights. The two lowest saliency weights are removed. The remaining weights
can be compensated using an inverse-Hessian update. When removing weight $w_j$, the
kept weights are updated by

$$
w_k \leftarrow w_k - w_j
\frac{(H^{-1})_{kj}}{(H^{-1})_{jj}} .
$$

This task uses a NumPy implementation of this local pruning procedure.

## Task

Implement `sparsegpt_2_4(W, X)`:

```python
def sparsegpt_2_4(W: np.ndarray, X: np.ndarray):
    ...
```

Inputs:

- `W` is a float matrix of shape $(m,n)$ representing a linear layer.
- `X` is a float matrix of shape $(s,n)$ representing calibration inputs.

The function must return a tuple:

```python
mask, W_hat
```

where:

- `mask` is an integer array with the same shape as `W`. A value of `1` means the
  weight is kept and `0` means it is pruned.
- `W_hat` is the compensated pruned weight matrix as `float64`.

The number of columns $n$ is a multiple of four. For each row of $W`, process each
group of four consecutive columns independently and keep exactly two entries.

The Hessian damping constant used by the grader is

$$
\lambda = 10^{-4}.
$$

Use NumPy operations. The output must match the reference algorithm.

## Example

```python
import numpy as np

W = np.array([[1., 2., 3., 4.]])
X = np.array([[1., 0., 1., 0.],
              [0., 1., 0., 1.]])

mask, W_hat = sparsegpt_2_4(W, X)

# mask contains two ones and two zeros.
# W_hat has zeros at the pruned locations and compensates kept values.
```

## What the gate checks

The gate builds its own NumPy oracle. It computes the damped Hessian, selects the
2:4 mask using Hessian saliency, applies inverse-Hessian compensation, and compares
the submitted result.

`mask_exact` requires the selected sparsity pattern to exactly match the oracle.
`rel_err` measures the relative error between the returned compensated weights and
the oracle weights:

$$
\mathrm{rel\_err} =
\frac{\lVert W_{\mathrm{hat}} - W_{\mathrm{oracle}}\rVert}
{\lVert W_{\mathrm{oracle}}\rVert + 10^{-12}} .
$$
