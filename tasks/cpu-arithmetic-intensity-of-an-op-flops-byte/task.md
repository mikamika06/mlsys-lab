## Context

The *arithmetic intensity* of a kernel is the ratio of floating‑point operations (FLOPs) to bytes moved from memory. It is a key quantity in the Roofline model, which predicts attainable performance on a given architecture by comparing compute capability with memory bandwidth.

For a naive matrix multiplication
$$C = A \times B,$$
with $A\in\mathbb{R}^{m\times k}$, $B\in\mathbb{R}^{k\times n}$ and $C\in\mathbb{R}^{m\times n}$, each output element requires $2\,k$ FLOPs (one multiply and one add). Hence the total number of FLOPs is
$$F = 2\,m\,n\,k.$$

Each element of $A$, $B$ and $C$ must be read or written once. If each element occupies $\ell$ bytes, the total amount of data moved is
$$D = (m k + k n + m n)\,\ell.$$

The arithmetic intensity is therefore
$$\mathrm{AI} = \frac{F}{D}
= \frac{2\,m\,n\,k}{(m k + k n + m n)\,\ell}.$$

## Task

Implement the function `arithmetic_intensity` that computes this quantity for a given matrix multiplication kernel.

```python
def arithmetic_intensity(m: int, n: int, k: int, elem_bytes: int = 8) -> float:
    """
    Return the arithmetic intensity (FLOPs per byte moved) of a naive
    matrix multiplication C = A @ B with shapes (m, k), (k, n).
    ``elem_bytes`` is the size in bytes of each scalar element.
    The result must be a Python float.
    """
    ...
```

The function should handle any positive integer dimensions and return a floating‑point value.

## Example

```python
>>> arithmetic_intensity(100, 200, 50)
7.142857142857143
```

(The numerator is $2\times100\times200\times50 = 2\,000\,000$ FLOPs; the denominator is $(100\times50 + 50\times200 + 100\times200)\times8 = 280\,000$ bytes.)

## What the gate checks

The grader evaluates the relative error of your result against a reference implementation. The metric `rel_err` must be $\le 10^{-9}$ for all test cases.
