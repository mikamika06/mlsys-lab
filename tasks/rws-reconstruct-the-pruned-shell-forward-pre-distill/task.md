## Context

Structured pruning removes selected input and output dimensions from a neural network layer. A pruned shell must preserve the forward behavior of the teacher network restricted to the dimensions that remain.

For a linear layer with weights $W \in \mathbb{R}^{m \times n}$, bias $b \in \mathbb{R}^{m}$, and input $x \in \mathbb{R}^{n}$, the teacher output is

$$
y = Wx + b .
$$

If output rows in `keep_rows` and input columns in `keep_cols` are retained, the pruned layer uses

$$
W' = W[\text{keep\_rows}, \text{keep\_cols}],
$$

$$
b' = b[\text{keep\_rows}],
$$

and the input restricted to the retained shell is

$$
x' = x[\text{keep\_cols}].
$$

The reconstructed forward pass is therefore

$$
y' = W'x' + b'.
$$

This operation is used before distillation, where the smaller model learns from the teacher restricted to the surviving dimensions.

## Task

Implement `pruned_shell_forward(W, b, x, keep_rows, keep_cols)`.

The arguments are NumPy arrays:

```python
def pruned_shell_forward(
    W: np.ndarray,
    b: np.ndarray,
    x: np.ndarray,
    keep_rows: np.ndarray,
    keep_cols: np.ndarray,
) -> np.ndarray:
    ...
```

`W` has shape $(m, n)$, `b` has shape $(m,)$, and `x` has shape $(n,)$.

`keep_rows` and `keep_cols` contain integer indices of the dimensions retained by pruning. Return the forward output of the pruned shell as a `float64` NumPy array.

The implementation should perform the slicing and matrix multiplication directly. Do not compute the full teacher output and then select values.

## Example

```python
import numpy as np

W = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
b = np.array([0.5, -1.0])
x = np.array([10, 20, 30], dtype=np.float64)

rows = np.array([1])
cols = np.array([0, 2])

y = pruned_shell_forward(W, b, x, rows, cols)
# array([132.])

# (4 * 10 + 6 * 30) - 1 = 179
# after keeping rows=[1], the expected value is:
# array([219.])
```

## What the gate checks

The gate builds several teacher layers, keep-sets, and inputs. It computes the oracle by slicing the teacher weights, bias, and input, then applying the pruned linear forward equation.

The returned result is compared with the oracle using the maximum absolute error

$$
\max_i |y'_i - y_i|.
$$

The value must be below $10^{-6}$. Implementations that slice the wrong axis, forget the bias, or use the unpruned input dimensions will fail.
