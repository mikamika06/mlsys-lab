## Context

In linear algebra a system of equations $U\,x = b$ with an upper‑triangular matrix $U \in \mathbb{R}^{n\times n}$ can be solved efficiently by *back substitution*.  
Because all entries below the main diagonal are zero, we can determine the unknowns starting from the last row and moving upwards:

$$
x_i = \frac{b_i - \sum_{j=i+1}^{n-1} U_{ij}\,x_j}{U_{ii}}, \qquad i=n-1,\dots,0 .
$$

This requires only $O(n^2)$ arithmetic operations compared to the $O(n^3)$ cost of a general Gaussian elimination.

## Task

Implement `back_substitution(U, b)`:

```python
def back_substitution(U: np.ndarray, b: np.ndarray) -> np.ndarray:
    ...
```

`U` is an $n \times n$ NumPy array that is upper‑triangular.  
`b` is a one‑dimensional array of length $n$.  
The function must return the solution vector `x` as a NumPy array of dtype `float64`.

## Example

```python
import numpy as np
U = np.array([[2, -1, 0],
              [0, 3, 4],
              [0, 0, 5]], dtype=np.float64)
b = np.array([1, 2, 3], dtype=np.float64)

x = back_substitution(U, b)
print(x)   # [ 1.33333333 -0.66666667  0.6       ]
```

## What the gate checks

The returned vector is compared to NumPy’s reference solver `np.linalg.solve`.  
The global relative L2 error must satisfy $\mathrm{rel\_err} \le 10^{-10}$.
