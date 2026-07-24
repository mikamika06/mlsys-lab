## Context

Neural network layers often compute a matrix multiplication followed by an
epilogue that adjusts and transforms the result. A common pattern is

$$
C = \mathrm{act}(A B + b),
$$

where $A \in \mathbb{R}^{m \times k}$ is an activation matrix, $B \in
\mathbb{R}^{k \times n}$ is a weight matrix, $b \in \mathbb{R}^{n}$ is a bias
vector, and $\mathrm{act}$ is an elementwise activation function.

Production kernels frequently fuse the bias addition and activation into the
matmul epilogue. The output values are written directly after the multiply
accumulate operation instead of materializing a separate intermediate matrix
for the bias and activation stages.

For ReLU activation,

$$
\mathrm{ReLU}(x) = \max(0, x).
$$

The fused computation produces

$$
C_{ij} = \max(0, \sum_{r=1}^{k} A_{ir}B_{rj} + b_j).
$$

## Task

Implement `fused_matmul_epilogue(A, B, bias, activation, out)`:

```python
def fused_matmul_epilogue(
    A: np.ndarray,
    B: np.ndarray,
    bias: np.ndarray,
    activation: str,
    out: np.ndarray,
) -> np.ndarray:
    ...
```

Compute the matrix multiplication epilogue and write the final result into the
provided `out` array. Return the same `out` object.

Inputs have shapes `(m, k)`, `(k, n)`, `(n,)`, and `(m, n)`. The supported
activation values are:

- `"relu"`: apply $\max(0, x)$.
- `"identity"`: leave values unchanged.

The implementation should avoid returning a separately allocated
post-matmul result. The goal is to model a fused epilogue where the final
destination buffer receives the completed values.

## Example

```python
import numpy as np

A = np.array([[1.0, -2.0]])
B = np.array([[3.0], [4.0]])
bias = np.array([1.0])
out = np.empty((1, 1))

result = fused_matmul_epilogue(A, B, bias, "relu", out)

# A @ B + bias = [[-4.0]]
# ReLU output is [[0.0]]
# result is out
```

## What the gate checks

The gate computes the expected values using a NumPy oracle:

$$
\mathrm{oracle} = \mathrm{activation}(A B + b).
$$

The returned values must match the oracle within a floating point tolerance.
The gate also verifies the epilogue contract by checking that the function
returns the provided `out` buffer rather than a separate post-matmul array.
