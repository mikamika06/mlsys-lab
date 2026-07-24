## Context

Large neural network matrices often contain a small number of columns with much
larger magnitudes than the rest. Quantizing all columns to int8 can introduce
large errors because these outlier columns dominate the matrix product.

A mixed-precision decomposition separates the weight matrix $W$ into regular
columns and outlier columns. Let $S$ be the set of outlier column indices and
let $\bar{S}$ be the remaining indices. The matrix product is approximated as

$$
XW \approx \operatorname{deq}(X_{\bar{S},8} W_{\bar{S},8})
+
X_{S,16} W_{S,16}.
$$

The first term uses int8 quantization with per-tensor scales:

$$
\operatorname{deq}(A_8B_8) = (A_8B_8) \cdot s_A \cdot s_B,
$$

where $A_8$ and $B_8$ are int8 tensors and $s_A$, $s_B$ are the corresponding
quantization scales. The outlier columns are kept in float16 so that their
contribution preserves more precision.

The goal is to reproduce this decomposition rather than simply multiplying the
full matrices in float32.

## Task

Implement:

```python
def mixed_precision_matmul(X: np.ndarray, W: np.ndarray, outlier_cols: np.ndarray) -> np.ndarray:
    ...
```

The inputs are:

- `X`: a float32 matrix of shape $(m, k)$.
- `W`: a float32 matrix of shape $(k, n)$.
- `outlier_cols`: a 1-D integer NumPy array containing column indices of $W$ that
  must remain in float16.

Return an array of shape $(m, n)$ containing the mixed-precision result.

Use this algorithm:

1. Split `W` into non-outlier columns and outlier columns.
2. Split `X` into the corresponding row-compatible matrices:
   the regular path uses all columns of `X`, while the outlier path uses the
   columns of `W` selected by `outlier_cols`.
3. Quantize the regular path inputs to int8 using symmetric max-absolute scaling.
4. Compute the dequantized int8 matrix product.
5. Compute the outlier product using float16 inputs and weights.
6. Add the two results into a float32 output matrix.

The quantization rule for a tensor $A$ is:

$$
s = \frac{\max(|A|)}{127}, \qquad A_8 = \operatorname{round}(A/s).
$$

If $A$ is all zeros, use $s = 1$.

## Example

```python
import numpy as np

X = np.array([[1.0, 2.0, 100.0]], dtype=np.float32)
W = np.array([
    [0.5, 1.0],
    [0.5, 2.0],
    [3.0, 4.0],
], dtype=np.float32)

outliers = np.array([1])

Y = mixed_precision_matmul(X, W, outliers)
```

The second weight column is computed with the higher precision outlier path.

## What the gate checks

The grader builds matrices with a few large-magnitude outlier columns and computes
the expected decomposition independently with NumPy. The returned matrix must
have relative error

$$
\frac{\lVert Y - Y_{\mathrm{oracle}}\rVert}
{\lVert Y_{\mathrm{oracle}}\rVert + 10^{-12}}
\leq 10^{-4}.
$$

The grader also computes a pure int8 quantized baseline. The submitted
implementation must be closer to the float32 matrix product than this baseline,
so the outlier path must provide a measurable accuracy improvement.
