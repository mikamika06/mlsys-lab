## Context

Magnitude pruning removes small‑weight entries from a tensor while keeping the largest ones.  
In *2:4* sparsity, every consecutive block of four elements in each row keeps only the two with the greatest absolute value; the other two are set to zero.  

The reconstruction error introduced by this constraint is measured by the mean squared error (MSE) between the original dense weight matrix $W$ and its pruned version $\hat W$:

$$
\mathrm{MSE}(W,\hat W)=\frac{1}{mn}\sum_{i=1}^{m}\sum_{j=1}^{n}\bigl(W_{ij}-\hat W_{ij}\bigr)^2,
$$

where $m\times n$ is the shape of $W$.

## Task

Implement a function that returns this MSE for a given weight matrix:

```python
def magnitude_prune_mse(W: np.ndarray) -> float:
    """
    Return the mean squared error between W and its 2:4 magnitude‑pruned version.
    The input is a 2‑D NumPy array of arbitrary shape; the output must be a Python
    float (not a NumPy scalar).
    """
```

The function should perform the pruning exactly as described above, using only NumPy operations.  
It must work for any number of rows and columns, including shapes that are not multiples of four.

## Example

```python
import numpy as np
W = np.array([[ 1., -2.,  3., -4.,  5., -6.],
              [ 7., -8.,  9., -10., 11., -12.]])
mse_val = magnitude_prune_mse(W)
print(mse_val)   # ≈ 0.16666666666666666
```

The pruned matrix keeps the two largest‑magnitude elements in each block of four, so the error is the average squared difference between $W$ and that pruned matrix.

## What the gate checks

The grader computes a reference MSE using a NumPy oracle implementation.  
It then compares the student's returned value to this reference with a relative error tolerance of $10^{-9}$.  
If the relative error exceeds this threshold, the submission fails the gate.
