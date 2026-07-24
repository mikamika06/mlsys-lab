## Context

Vector-wise (a.k.a. row/column-wise) int8 quantization speeds up
$Y = XW$ by giving every row of $X$ and every column of $W$ its own int8
scale, instead of one scale for the whole tensor. For a row
$x \in \mathbb{R}^d$ of $X \in \mathbb{R}^{n\times d}$ and a column
$w \in \mathbb{R}^d$ of $W \in \mathbb{R}^{d\times m}$:

$$
s_x = \frac{\max_j |x_j|}{127}, \qquad s_w = \frac{\max_i |w_i|}{127}.
$$

Both operands are quantized to signed 8-bit integers with those scales:

$$
X^{q}_{ij} = \mathrm{round}\!\left(\frac{X_{ij}}{s_{x,i}}\right), \qquad
W^{q}_{jk} = \mathrm{round}\!\left(\frac{W_{jk}}{s_{w,k}}\right),
$$

clipped to $[-127, 127]$. The int8 matrices are multiplied with int32
accumulation, then the single result is dequantized in one shot using the
**outer product** of the per-row and per-column scales — this is the whole
point of vector-wise quantization: one int8 GEMM produces a result that
needs only an elementwise rescale, not a full requantization:

$$
Y_{ik} = \Big(\sum_j X^{q}_{ij} W^{q}_{jk}\Big)\cdot s_{x,i}\, s_{w,k}
       = \big(X^{q} W^{q}\big)_{ik} \cdot (s_x \otimes s_w)_{ik}.
$$

## Task

Implement `vector_wise_int8_matmul(X, W)`:

```python
import numpy as np

def vector_wise_int8_matmul(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    ...
```

- `X`: `(n, d)` float64.
- `W`: `(d, m)` float64.

Follow the pipeline above exactly: per-row scale for `X`, per-column scale
for `W`, round-and-clip to `int8` range, integer matmul, then dequantize
by the outer product of the two scale vectors. Return a `(n, m)` float64
array `Y`.

## Example

```python
import numpy as np
X = np.array([[1.0, -2.0, 3.0]])
W = np.array([[0.5], [-1.5], [2.0]])
Y = vector_wise_int8_matmul(X, W)
# s_x = 3.0/127, s_w = 2.0/127 (per the single row / single column here)
# Xq = round(X / s_x) = round([42.33, -84.67, 127.0]) = [42, -85, 127]
# Wq = round(W / s_w) = round([31.75, -95.25, 127.0]) = [32, -95, 127]
# Xq @ Wq = 42*32 + (-85)*(-95) + 127*127 = 1344 + 8075 + 16129 = 25548
# Y = 25548 * (3.0/127) * (2.0/127) ≈ 9.5039
```

## What the gate checks

The grader loads committed fixtures `int8_x.npy` (`(32, 64)`) and
`int8_w.npy` (`(64, 24)`) — moderate, no-strong-outlier magnitudes so
neither operand collapses onto a degenerate scale — and runs the exact
same vector-wise int8 pipeline independently in NumPy to get the oracle's
`Y`.

The gate metric is `rel_err`, the global relative L2 error between your
`Y` and the oracle's `Y`:

$$
\text{rel\_err} = \frac{\lVert Y_{\text{yours}} - Y_{\text{oracle}}\rVert_2}{\lVert Y_{\text{oracle}}\rVert_2 + 10^{-12}} < 10^{-3}.
$$

Because both sides run the *same* deterministic quantization algorithm,
a correct implementation reproduces the oracle almost exactly — the
threshold only needs to absorb integer-rounding tie-break edge cases.
Quantizing per-tensor instead of per-row/per-column, dequantizing with
`s_x + s_w` or `s_x * s_w` (broadcast to the wrong shape) instead of the
outer product, or skipping the round/clip step will all miss the gate by
a wide margin.
