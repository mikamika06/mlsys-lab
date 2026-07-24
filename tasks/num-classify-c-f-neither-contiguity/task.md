## Context

In NumPy, the memory layout of an array is described by its **strides** and the flags `C_CONTIGUOUS` and `F_CONTIGUOUS`.  
An array is *C contiguous* if its elements are stored in row‑major order; it is *F contiguous* if they are stored in column‑major order.  A one‑dimensional array, or an array that happens to be both row‑major and column‑major (e.g., a single column), satisfies both flags.

The flags can be inspected via `arr.flags.c_contiguous` and `arr.flags.f_contiguous`.  These boolean attributes are the canonical way to determine contiguity in NumPy.

## Task

Implement the function

```python
def classify_contiguity(arr: np.ndarray) -> str:
    ...
```

It receives a two‑dimensional NumPy array and returns one of three strings:

* `"C"` – the array is C contiguous **and not** F contiguous,
* `"F"` – the array is F contiguous **and not** C contiguous,
* `"Neither"` – any other case (including when both flags are true or neither).

The function must use only NumPy’s public API and should work for arrays of arbitrary dtype.

## Example

```python
import numpy as np

# C contiguous
A = np.arange(12).reshape((3, 4))
print(classify_contiguity(A))          # "C"

# F contiguous
B = A.T
print(classify_contiguity(B))          # "F"

# Neither (slicing creates a non‑contiguous view)
C = A[::2]
print(classify_contiguity(C))          # "Neither"

# 1‑D array – both flags true, classified as "Neither"
D = np.arange(5)
print(classify_contiguity(D.reshape((5, 1))))  # "Neither"
```

## What the gate checks

The grader compares your output against a reference implementation that uses NumPy’s `flags` attributes.  
A single metric **exact_match** is used: it must equal `1.0`.  Any deviation (including wrong string or type) causes failure.
