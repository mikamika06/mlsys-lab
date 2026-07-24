## Context

The condition number of a matrix measures how much numerical errors can be amplified
when solving linear systems. For a matrix $A$, the 2-norm condition number is

$$
\kappa_2(A) = \lVert A \rVert_2 \lVert A^{-1} \rVert_2 .
$$

The Hilbert matrix is a classic example of a matrix that becomes increasingly
ill-conditioned as its size grows. The $n \times n$ Hilbert matrix is defined by

$$
H_{ij} = \frac{1}{i+j+1},
$$

where indices start at $i,j=0$. As $n$ increases, small floating point errors can
become much larger after computations involving $H_n$.

The condition number can be computed numerically with NumPy as

$$
\kappa_2(H_n) = \frac{\sigma_{\max}(H_n)}{\sigma_{\min}(H_n)},
$$

where $\sigma_{\max}$ and $\sigma_{\min}$ are the largest and smallest singular
values.

## Task

Implement `hilbert_condition_numbers(ns)`:

```python
def hilbert_condition_numbers(ns):
    ...
```

The function receives an iterable of positive integer matrix sizes. For every
size $n$, construct the Hilbert matrix $H_n$ using NumPy operations and return a
1-D NumPy array of `float64` values containing

$$
\log_{10}(\kappa_2(H_n)).
$$

The returned array must have the same length and order as `ns`.

## Example

```python
import numpy as np

values = hilbert_condition_numbers([2, 4, 8])

# values contains approximately:
# [0.77815125, 2.26265531, 5.69512874]
```

## What the gate checks

The gate computes a reference result using NumPy's singular value decomposition on
the Hilbert matrices. It compares the returned log10 condition numbers against
that reference using the mean relative error:

$$
\mathrm{rel\_err}
=
\frac{1}{m}
\sum_{i=1}^{m}
\frac{|x_i-y_i|}{|y_i|+10^{-12}} .
$$

The score must satisfy $\mathrm{rel\_err} \le 0.05$.
