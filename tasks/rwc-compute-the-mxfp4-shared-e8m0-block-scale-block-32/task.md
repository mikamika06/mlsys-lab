## Context

In the MXFP4 quantisation format used by OCP‑MX, a contiguous block of 32 weight values shares a single power‑of‑two scaling factor.  
The underlying *e2m1* representation can encode signed integers in the range $[-6,\;6]$.  
To fit all 32 values into this range we choose an integer exponent $e$ such that

$$
|w_i|\;2^{-e}\;\leq\;6 \quad\text{for every }i=1,\dots,32 .
$$

The smallest non‑negative integer $e$ satisfying the inequality is used for the whole block.  
This exponent is returned as a 32‑bit signed integer.

## Task

Implement `compute_shared_e8m0_scale(weights)`:

```python
def compute_shared_e8m0_scale(weights):
    ...
```

`weights` is a list of lists of floats of shape $(B,\;32)$ where $B$ is the number of blocks.  
The function must return a one‑dimensional integer array of length $B$, containing the exponent for each block.

The implementation should be fully vectorised; no explicit Python loops over the blocks or elements are required.

## Example

```python
weights = [
    [0, 1.5, -2.3, 6.0] + [0]*28,
    [7.0, 0, 0, 0] + [0]*28
]
exponents = compute_shared_e8m0_scale(weights)
print(exponents)   # [0, 1]
```

Explanation:  
For the first block the maximum absolute value is $6$, so $e=\lceil\log_2(6/6)\rceil=0$.  
For the second block $\max|w|=7>6$ and $\lceil\log_2(7/6)\rceil=1$.

## What the gate checks

The grader computes a reference exponent for each block using Python’s `log2` and `ceil`.  
Your output must match this reference exactly (element‑wise equality).  No tolerance is allowed.  
If any element differs, or if the shape/dtype is incorrect, the solution fails.
