## Context

Low-bit inference often uses a rotation before quantization to make values easier to represent. A fixed Hadamard rotation is cheap and deterministic, while a learned orthogonal rotation $R$ can reduce quantization error.

For a matrix $Z$, symmetric 4-bit quantization maps values using

$$
Q_4(Z) = \operatorname{clip}\left(\operatorname{round}\left(\frac{Z}{s}\right), -8, 7\right)s,
$$

where $s = \frac{\max(|Z|)}{7}$.

Given weights $W$ and activations $X$, the quantized output approximation is compared with the full precision output

$$
Y = WX.
$$

A rotation $R$ with $R^TR = I$ can be inserted without changing the mathematical operation:

$$
WX = (WR)(R^TX).
$$

The quantized approximation becomes

$$
\hat{Y}_R = Q_4(WR)Q_4(R^TX).
$$

A Hadamard transform is a fixed orthogonal transform. A learned rotation is expected to produce equal or lower quantization error because it was optimized for the specific tensors.

## Task

Implement `w4a4_rotation_mse(W, X, R)`:

```python
def w4a4_rotation_mse(W: np.ndarray, X: np.ndarray, R: np.ndarray) -> tuple[float, float]:
    ...
```

The function receives a weight matrix $W$, an activation matrix $X$, and a learned orthogonal rotation $R$.

Return two values:

1. The mean squared error between $WX$ and the W4A4 approximation using a normalized Hadamard rotation.
2. The mean squared error between $WX$ and the W4A4 approximation using the provided rotation $R$.

Use NumPy operations only. The quantization must use the symmetric 4-bit rule from the context. The Hadamard matrix is constructed recursively and normalized so that it is orthogonal.

## Example

```python
import numpy as np

W = np.eye(4)
X = np.arange(16, dtype=np.float64).reshape(4, 4)
R = np.eye(4)

mse_h, mse_r = w4a4_rotation_mse(W, X, R)
```

The returned values are floating point mean squared errors. The second value is not required to always be smaller for arbitrary inputs, but the grading tensors provide a learned rotation where it is lower than the Hadamard baseline.

## What the gate checks

The gate computes the same W4A4 algorithm independently as a NumPy oracle and compares both returned errors. The reported metric is the largest absolute difference between the candidate and oracle values, plus a penalty if the learned rotation error is larger than the Hadamard error.

The final metric must satisfy

$$
\mathrm{mse} \le 10^{-6}.
$$
