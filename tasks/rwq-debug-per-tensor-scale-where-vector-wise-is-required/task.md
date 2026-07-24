## Context

Quantized matrix multiplication replaces floating point values with integer values
and stores scale factors separately. For a matrix $X \in \mathbb{R}^{m \times k}$,
a common vector-wise activation quantization uses one scale per row:

$$
s^X_i = \frac{\max_j |X_{ij}|}{127}.
$$

The quantized activation is

$$
X^q_{ij} = \operatorname{round}\left(\frac{X_{ij}}{s^X_i}\right),
$$

with values clipped to the int8 range.

For a weight matrix $W \in \mathbb{R}^{k \times n}$, a per-output-channel
quantization uses one scale per column:

$$
s^W_j = \frac{\max_i |W_{ij}|}{127},
$$

and

$$
W^q_{ij} = \operatorname{round}\left(\frac{W_{ij}}{s^W_j}\right).
$$

The reconstructed matrix multiplication is computed as

$$
Y_{ij} \approx \left(X^q W^q\right)_{ij} s^X_i s^W_j .
$$

Using a single scale for the entire tensor loses information when different rows
or columns have different magnitudes.

## Task

Implement `int8_matmul_per_channel(X, W)`:

```python
def int8_matmul_per_channel(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    ...
```

The function receives a floating point activation matrix `X` with shape $(m,k)$
and a floating point weight matrix `W` with shape $(k,n)$. Return the
float64 matrix $Y$ with shape $(m,n)$ using vector-wise int8 quantization:

1. Compute one int8 scale for each row of `X`.
2. Compute one int8 scale for each column of `W`.
3. Quantize both matrices with rounding and clipping to `[-127,127]`.
4. Perform integer matrix multiplication.
5. Dequantize with the row and column scale vectors.

Use NumPy operations only. Do not use Python loops over matrix elements.

## Example

```python
import numpy as np

X = np.array([[1.0, 2.0], [10.0, 1.0]])
W = np.array([[1.0, 3.0], [2.0, 1.0]])

Y = int8_matmul_per_channel(X, W)
```

The result should approximate the floating point product:

$$
\begin{bmatrix}
1 & 2\\
10 & 1
\end{bmatrix}
\begin{bmatrix}
1 & 3\\
2 & 1
\end{bmatrix}
=
\begin{bmatrix}
5 & 5\\
12 & 31
\end{bmatrix}.
$$

## What the gate checks

The gate computes a NumPy reference implementation of vector-wise int8
quantization and compares the submitted result using global relative error:

$$
\mathrm{rel\_err} =
\frac{\lVert Y_{\mathrm{candidate}} - Y_{\mathrm{reference}} \rVert_2}
{\lVert Y_{\mathrm{reference}} \rVert_2 + 10^{-12}} .
$$

The result must satisfy $\mathrm{rel\_err} \le 10^{-3}$. A solution that uses
one global tensor scale for `X` or `W` loses accuracy on inputs with uneven row
and column magnitudes and will fail.
