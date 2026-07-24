## Context

A linear layer computes a matrix product

$$Y = W X,$$

where $W \in \mathbb{R}^{m \times n}$ contains weights and $X \in \mathbb{R}^{n \times b}$ contains activations.

A channel-wise migration can move a diagonal scaling matrix from one side of the product to the other. For a nonzero scale vector $s \in \mathbb{R}^n$, define

$$
S = \operatorname{diag}(s).
$$

The identity

$$
W X = (W S)(S^{-1} X)
$$

allows a system to store a transformed weight matrix and a transformed activation matrix while preserving the exact linear output.

This transformation is useful in quantization pipelines because the activation values can have a smaller dynamic range after dividing channels by $s$:

$$
X_{\mathrm{fold}} = \operatorname{diag}(1/s) X .
$$

The original activation range compared with the folded range can be measured as

$$
\frac{\max |X|}{\max |X_{\mathrm{fold}}|}.
$$

## Task

Implement `fold_diag_scales(W, X, s)`:

```python
def fold_diag_scales(
    W: np.ndarray, X: np.ndarray, s: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    ...
```

The inputs are:

- `W`: a floating point array with shape $(m, n)$.
- `X`: a floating point array with shape $(n, b)$.
- `s`: a nonzero floating point vector with shape $(n,)$.

Return four values:

1. `W_fold`, equal to $W \operatorname{diag}(s)$.
2. `X_fold`, equal to $\operatorname{diag}(1/s)X$.
3. `Y_fold`, equal to `W_fold @ X_fold`.
4. `range_reduction_ratio`, equal to $\max|X| / \max|X_{\mathrm{fold}}|$.

Use NumPy operations. Do not explicitly build diagonal matrices.

## Example

```python
import numpy as np

W = np.array([[2.0, 3.0], [4.0, 5.0]])
X = np.array([[1.0, 2.0], [3.0, 4.0]])
s = np.array([3.0, 0.5])

W_fold, X_fold, Y_fold, ratio = fold_diag_scales(W, X, s)

# W_fold @ X_fold is equal to W @ X
```

## What the gate checks

The gate builds NumPy oracle computations for the diagonal migration identity. It checks that the submitted folded product matches the oracle product within a small maximum absolute error and that the reported range-reduction statistic is computed from the folded activation matrix.
