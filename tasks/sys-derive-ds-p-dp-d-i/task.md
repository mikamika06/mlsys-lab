## Context

In attention, the softmax matrix $P$ is computed row by row from scores $S$:

$$
P_i = \mathrm{softmax}(S_i).
$$

During the backward pass, the gradient with respect to the scores can be derived without explicitly constructing the full softmax Jacobian. For an upstream gradient $dP_i$, the Jacobian-vector product of softmax gives

$$
dS_i = P_i \circ \left(dP_i - \sum_j P_{ij}dP_{ij}\right),
$$

where $\circ$ denotes element-wise multiplication.

The FlashAttention backward algorithm uses this form because it avoids materializing large Jacobian matrices. The term

$$
D_i = \sum_j P_{ij}dP_{ij}
$$

is a scalar for each row and is broadcast across the row during subtraction.

## Task

Implement `derive_ds(P, dP)`:

```python
def derive_ds(P: np.ndarray, dP: np.ndarray) -> np.ndarray:
    ...
```

The function receives two arrays with shape $(n, m)$:

- `P` is a row-wise softmax probability matrix.
- `dP` is the upstream gradient of the same shape.

Return the score gradient $dS$ using the derived expression:

$$
dS = P \circ (dP - D),
$$

where $D$ is the column-vector of row reductions

$$
D_i = \sum_j P_{ij}dP_{ij}.
$$

The output must be a `float64` NumPy array with shape $(n, m)$. Use vectorized NumPy operations.

## Example

```python
import numpy as np

P = np.array([[0.25, 0.75], [0.5, 0.5]])
dP = np.array([[2.0, 4.0], [3.0, 1.0]])

dS = derive_ds(P, dP)
# [[-0.375,  0.375],
#  [ 0.5,   -0.5 ]]
```

## What the gate checks

The gate builds several attention-shaped inputs and computes the oracle result using the explicit softmax Jacobian-vector product:

$$
J_i = \mathrm{diag}(P_i) - P_iP_i^\top,
$$

then

$$
dS_i = J_i dP_i.
$$

The returned value is compared against this NumPy oracle using relative error:

$$
\mathrm{rel\_err} =
\frac{\lVert dS_{\mathrm{candidate}}-dS_{\mathrm{oracle}}\rVert}
{\lVert dS_{\mathrm{oracle}}\rVert + 10^{-12}}.
$$

A result with $\mathrm{rel\_err} \le 10^{-5}$ passes.
