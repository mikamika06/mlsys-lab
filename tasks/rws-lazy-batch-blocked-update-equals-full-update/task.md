## Context

Quantization methods for neural network weights can reduce memory usage by replacing floating point values with a smaller set of representable values. SparseGPT-style methods compensate for quantization error by applying updates derived from an approximate inverse Hessian.

Let $W \in \mathbb{R}^{m \times n}$ be a weight matrix and $X \in \mathbb{R}^{n \times d}$ be calibration data. A Hessian approximation is

$$
H = XX^\top + 10^{-6}I .
$$

The inverse Hessian $H^{-1}$ determines how quantization error in one column is distributed into later columns. For column $j$, let $Q_j$ be the quantized column and

$$
e_j = W_j - Q_j .
$$

The correction applied after quantizing column $j$ is

$$
W_{:,j+1:}
\leftarrow
W_{:,j+1:}
-
\frac{e_j}{H^{-1}_{jj}}
(H^{-1}_{j,j+1:}) .
$$

A production implementation can process columns in blocks while keeping the same lazy update rule. The block size changes the grouping of work, but the final corrected weights should match the reference column-by-column update.

## Task

Implement `lazy_batch_update(W, X, s, blocksize)`:

```python
def lazy_batch_update(
    W: np.ndarray,
    X: np.ndarray,
    s: int,
    blocksize: int,
) -> np.ndarray:
    ...
```

The function receives:

- `W`: a floating point matrix of shape $(m,n)$.
- `X`: calibration data of shape $(n,d)$.
- `s`: the symmetric quantization range.
- `blocksize`: the number of columns grouped for processing.

Return the final corrected matrix as `float64`.

Use this quantizer:

$$
\Delta = \frac{\max(|A|)}{s},
$$

$$
Q(A)=\operatorname{clip}(\operatorname{round}(A/\Delta),-s,s)\Delta .
$$

If $\Delta=0$, return zeros.

The returned result must match the full column update even when `blocksize` is greater than one.

## Example

```python
import numpy as np

W = np.array([[1.2, -0.8, 0.4],
              [0.3, 1.1, -1.4]])

X = np.eye(3)

Y = lazy_batch_update(W, X, 4, 2)

# Y matches the blocksize=1 reference update
```

## What the gate checks

The gate computes its own NumPy reference by running the full column-by-column lazy update algorithm.

The submitted result is compared with the oracle using

$$
\mathrm{rel\_err}
=
\frac{\lVert Y_{\mathrm{candidate}}-Y_{\mathrm{oracle}}\rVert_2}
{\lVert Y_{\mathrm{oracle}}\rVert_2+10^{-12}} .
$$

The metric must satisfy $\mathrm{rel\_err}<10^{-6}$.
