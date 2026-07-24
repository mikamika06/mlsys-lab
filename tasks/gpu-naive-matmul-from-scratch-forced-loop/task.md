## Context

Matrix multiplication is a core primitive in linear algebra.  
Given two matrices $A \in \mathbb{R}^{m\times k}$ and $B \in \mathbb{R}^{k\times n}$, the product $C = AB$ has entries  

$$
C_{ij} = \sum_{p=1}^{k} A_{ip}\; B_{pj},
$$

for all rows $i$ of $A$ and columns $j$ of $B$.  
The most straightforward way to compute $C$ is by evaluating this triple sum directly, i.e. with three nested loops.

In a high‑performance implementation one would use SIMD instructions or GPU kernels, but here we focus on the pure algorithmic structure that can be expressed in plain Python and NumPy.

## Task

Implement the function `matmul_loops(A, B)`:

```python
def matmul_loops(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    ...
```

- `A` and `B` are 2‑D NumPy arrays of shape `(m, k)` and `(k, n)` respectively.
- The function must compute the matrix product by iterating explicitly over all indices with nested Python loops – **no** use of `np.dot`, `@`, or any other high‑level NumPy routine that performs the multiplication for you.
- Return a new NumPy array containing the result.  
  The returned array must have dtype `float64`.

The implementation should be clear, idiomatic, and produce numerically accurate results.

## Example

```python
import numpy as np

A = np.array([[1., 2.],
              [3., 4.]])
B = np.array([[5., 6.],
              [7., 8.]])

C = matmul_loops(A, B)
print(C)
```

Output:

```
[[19. 22.]
 [43. 50.]]
```

## What the gate checks

Two metrics are evaluated.

1. **Relative error** (`rel_err`) – the grader computes a reference result with `A @ B` and compares your output using  
   $$ \mathrm{rel\_err} = \frac{\|C_{\text{candidate}}-C_{\text{ref}}\|_F}
                               {\|C_{\text{ref}}\|_F}. $$
   The value must be **≤ $1\times10^{-12}$**.

2. **Line count** (`line_count`) – the grader inspects your source code for the number of executable lines (excluding comments and blank lines).  
   Your implementation must contain at least **5 lines** of code to discourage trivial wrappers that merely call `np.matmul`.

If either metric fails, the solution is rejected.

Note: using NumPy’s built‑in vectorized multiplication (`@`, `dot`) will cause a higher line count but also results in a different source pattern that the grader flags as invalid.  Write an explicit triple loop instead.
