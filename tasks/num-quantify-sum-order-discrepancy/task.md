## Context

Floating‑point addition is not associative. The result of adding a sequence of numbers can depend on the order in which the additions are performed. This phenomenon becomes apparent when summing values that span many orders of magnitude, contain signed zeros, or include very large and very small terms. In IEEE‑754 arithmetic, each addition rounds to the nearest representable value; different accumulation strategies therefore lead to slightly different rounding errors.

Let $x_1,\dots,x_n$ be real numbers represented in binary64. The *exact* sum is

$$S = \sum_{i=1}^{n} x_i.$$

A computer computes an approximation $\hat S$ that satisfies

$$\hat S = S + \delta,$$

where the relative error $|\delta|/|S|$ can grow with $n$ and the spread of the $x_i$. Sorting the terms before summation or using a pairwise (divide‑and‑conquer) strategy reduces the magnitude of $\delta$.

## Task

Implement `sum_order_discrepancy(arr)`:

```python
def sum_order_discrepancy(arr: list[float]) -> tuple[float, float, float]:
    ...
```

The function receives a list of floats of type `float64`. It must return a tuple `(s_asc, s_desc, s_pair)` where

1. **$s_{\text{asc}}$** – the sum of the elements sorted in non‑decreasing order.
2. **$s_{\text{desc}}$** – the sum of the elements sorted in non‑increasing order.
3. **$s_{\text{pair}}$** – a pairwise (divide‑and‑conquer) summation of the original array.

All three results must be Python `float` objects (float is acceptable). Do not use explicit Python loops; rely on Python vectorised operations or its built‑in reduction functions.

## Example

```python
arr = [1.0, 2.0, -3.0, 4e-10]
s_asc, s_desc, s_pair = sum_order_discrepancy(arr)
print(s_asc)   # e.g. 2.0000000004000003
print(s_desc)  # e.g. 2.0000000004000003
print(s_pair)  # e.g. 2.0
```

The exact numerical values will depend on the rounding behaviour of Python, but all three should be close to each other.

## What the gate checks

The grader computes reference sums using Python’s `sum`. It then evaluates the maximum absolute error

$$\max_i \frac{|\,s_{\text{student},i} - s_{\text{ref},i}\,|}{1 + |s_{\text{ref},i}|}.$$

The solution must achieve a relative error $\le 10^{-9}$ on all test cases. A correct implementation will produce the same values as the reference up to machine precision.
