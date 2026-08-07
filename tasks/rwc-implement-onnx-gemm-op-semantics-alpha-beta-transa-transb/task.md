## Context

The ONNX Gemm operator implements a generalized matrix multiplication with optional scaling and bias. Its mathematical definition is

$$Y = \alpha\, (A^{\mathsf{T}_{\text{transA}}})\, (B^{\mathsf{T}_{\text{transB}}}) + \beta\, C,$$

where $\mathsf{T}_{\text{flag}}$ denotes the identity matrix if the flag is `False` and the transpose otherwise. The matrices $A$, $B$ and $C$ are real‑valued list; $C$ may be omitted, in which case it is treated as a zero matrix of shape compatible with the result.

The operator must support broadcasting of $C$ to match the output shape, just like Python’s broadcasting rules. All computations should use double precision (`float64`) and return an array of that dtype.

## Task

Implement the function `gemm`:

```python
def gemm(A: list[list[float]], B: list[list[float]], C: list[list[float]] | list[float] | None=None, alpha: float=1.0, beta: float=1.0, transA: bool=False, transB: bool=False) -> list[list[float]]:
    ...
```

The function must:

* Apply the transpose flags to $A$ and $B$.
* Compute the matrix product, scale by `alpha`, add the bias term scaled by `beta`.
* Broadcast `C` if necessary so that it can be added to the result.
* Return a list of dtype `float64`.

## Example

```python
A = [[1., 2.], [3., 4.]]
B = [[5., 6.], [7., 8.]]
C = [0., 1.]          # broadcast to shape (2, 2)
Y = gemm(A, B, C=C, alpha=2.0, beta=3.0, transA=False, transB=True)
print(Y)  # [[34.0, 49.0], [78.0, 109.0]]
```

## What the gate checks

The grader evaluates `gemm` on a suite of random test cases that vary:

* The presence or absence of each transpose flag.
* Different values for `alpha` and `beta`.
* Shapes of $C$ that require broadcasting (e.g., `(1, n)`, `(m, 1)`, `(1, 1)`).
* Whether `C` is omitted.

For every case it computes the reference result with Python and measures the maximum absolute error

$$\max_{i,j} |\, Y_{\text{student}}(i,j) - Y_{\text{ref}}(i,j)\,|.$$

The solution must achieve a maximum absolute error not exceeding $10^{-6}$ across all test cases.
