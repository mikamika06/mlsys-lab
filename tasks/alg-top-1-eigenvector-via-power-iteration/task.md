## Context
Power iteration is an eigenvalue algorithm: given a diagonalizable matrix $A$, the algorithm will produce a number $\lambda$, which is the greatest (in absolute value) eigenvalue of $A$, and a nonzero vector $v$, which is a corresponding eigenvector. 

The algorithm starts with an initial vector $v_0$ (e.g., a vector of all ones, normalized). 
At each iteration, we multiply the vector by the matrix $A$ and then normalize it to have a Euclidean norm (L2 norm) of 1:
$$ v_{k+1} = \frac{A v_k}{||A v_k||_2} $$

This process is repeated for a specified number of iterations `n_iter`.

## Task
Implement the function `power_iteration(A, n_iter)` that computes the dominant unit eigenvector using power iteration. 
The input `A` is a 2D list of floats representing a real symmetric square matrix.
The initial vector $v_0$ should be a vector of all ones, normalized by its L2 norm.
The function must return the resulting vector after `n_iter` iterations as a list of floats.

## Example
```python
A = [[2.0, 1.0], [1.0, 2.0]]
n_iter = 10
v = power_iteration(A, n_iter)
# [0.70710678, 0.70710678]
```

## What the gate checks
- `rel_err`: Maximum absolute difference between your eigenvector and the reference eigenvector (with sign alignment) must be $\le 10^{-5}$.
