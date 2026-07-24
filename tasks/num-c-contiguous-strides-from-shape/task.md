## Context

In NumPy, an array’s memory layout is described by its **strides**.  
The stride of a dimension tells how many bytes one must skip to move to the next element along that axis.  
For a C‑contiguous (row‑major) array, the last dimension has the smallest stride equal to the item size of the dtype, and each preceding stride equals the product of the sizes of all following dimensions times the item size.

Mathematically, for shape \((n_0,n_1,\dots ,n_{k-1})\) and element size \(s\), the C‑contiguous strides are

$$
\text{stride}_i = s \prod_{j=i+1}^{k-1} n_j .
$$

NumPy exposes these values via the ``.strides`` attribute of an array.

## Task

Implement a function that, given a shape tuple and a dtype (or any object accepted by ``np.dtype``), returns the C‑contiguous byte strides as a tuple of integers.

```python
def c_contig_strides(shape: tuple[int, ...], dtype: np.dtype | str | type) -> tuple[int, ...]:
    ...
```

The function must not create an actual array; it should compute the strides purely from the shape and dtype information.  
The returned tuple must match NumPy’s ``np.empty(shape, dtype).strides``.

## Example

```python
import numpy as np
print(c_contig_strides((3, 4), np.int32))
# (16, 4)
```

Here the element size of ``int32`` is 4 bytes; the stride for the first dimension is \(4 \times 4 = 16\) and for the second it is 4.

## What the gate checks

The grader compares your result to NumPy’s reference using an exact match.  
Any discrepancy in any element causes the gate to fail.
