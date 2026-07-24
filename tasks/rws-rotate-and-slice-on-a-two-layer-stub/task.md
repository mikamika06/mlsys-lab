## Context

A two-layer linear network can contain a wide hidden dimension that is expensive to keep during inference. A common compression step rotates the hidden representation into a basis aligned with its calibration covariance, then keeps only the most important directions.

For a hidden activation matrix $H \in \mathbb{R}^{n \times m}$, compute the covariance

$$C = \frac{1}{n-1}(H-\bar{H})^\top(H-\bar{H}),$$

where $\bar{H}$ is the row mean. If $Q$ contains the eigenvectors of $C$ ordered by decreasing eigenvalue, then $HQ$ is the rotated representation. Keeping the first $k$ columns gives

$$H_k = H Q_{:,0:k}.$$

The following identity keeps the two-layer network consistent before slicing:

$$H W_2 = (H Q)(Q^\top W_2).$$

After slicing, only the first $k$ rotated channels and the corresponding rows of $Q^\top W_2$ are used.

## Task

Implement `rotate_and_slice(W1, b1, W2, b2, X_cal, X, k)`.

The inputs are NumPy arrays for two adjacent linear layers:

```python
def rotate_and_slice(
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
    X_cal: np.ndarray,
    X: np.ndarray,
    k: int,
) -> np.ndarray:
    ...
```

`W1` has shape $(d,m)$, `b1` has shape $(m,)$, `W2` has shape $(m,o)$, and `b2` has shape $(o,)`. `X_cal` is calibration input data and `X` is evaluation input data.

The function must:

1. Compute hidden calibration activations $H_{cal}=X_{cal}W_1+b_1$.
2. Compute the covariance eigenbasis $Q$ using `np.linalg.eigh`, ordered by descending eigenvalue.
3. Rotate the hidden output by $Q$ and the second layer input weights by $Q^\top$.
4. Slice the rotated hidden dimension to the first `k` channels.
5. Return the compressed end-to-end output for `X`.

The result must be a NumPy array of `float64`.

## Example

```python
import numpy as np

W1 = np.eye(3)
b1 = np.zeros(3)
W2 = np.arange(6, dtype=float).reshape(3, 2)
b2 = np.zeros(2)
X_cal = np.array([[1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])
X = np.array([[1., 1., 1.]])

Y = rotate_and_slice(W1, b1, W2, b2, X_cal, X, 2)
```

The output is the second layer prediction after rotating and removing one hidden direction.

## What the gate checks

The gate builds a two-layer linear model and computes the full-rank NumPy oracle output internally. It compares the submitted implementation against the mathematically equivalent rotation and slicing procedure.

The reported relative error is

$$\mathrm{rel\_err} =
\frac{\lVert Y_{candidate}-Y_{oracle}\rVert_2}
{\lVert Y_{oracle}\rVert_2+10^{-12}}.$$

The implementation passes when $\mathrm{rel\_err} \le 10^{-6}$. Implementations that slice the original hidden channels without applying the covariance rotation produce a different approximation and fail.
