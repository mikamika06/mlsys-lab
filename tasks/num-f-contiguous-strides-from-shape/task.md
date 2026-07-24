## Context

In NumPy an array’s *strides* describe how many bytes one must skip to move from one element to the next along a given axis. For a C‑contiguous layout the last axis changes fastest; for a Fortran (column‑major) layout the first axis changes fastest. If an array has shape $(n_0,n_1,\dots ,n_{d-1})$ and each element occupies $s$ bytes, then in Fortran order the stride along axis $i$ is

$$
\text{stride}_i = s \prod_{j=0}^{i-1} n_j,
$$

with the empty product equal to $1$. Thus for shape $(3,4,5)$ and a 64‑bit float ($s=8$) the strides are $(8,\;24,\;96)$.

## Task

Implement `f_contiguous_strides_from_shape(shape)`:

```python
def f_contiguous_strides_from_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    ...
```

It receives a shape as a tuple of positive integers and returns the corresponding Fortran‑contiguous strides in bytes. The function must work for any non‑negative dimensionality; an empty shape should yield `()`.

## Example

```python
>>> f_contiguous_strides_from_shape((3, 4, 5))
(8, 24, 96)
```

## What the gate checks

The grader compares your output to `np.zeros(shape, dtype=np.float64, order='F').strides`. The comparison is an exact match of tuples; any discrepancy causes failure. No other metrics are evaluated.
