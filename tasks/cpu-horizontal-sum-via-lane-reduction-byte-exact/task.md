## Context

SIMD (Single Instruction, Multiple Data) processors expose a vector register that holds several **lanes** of data. A *horizontal sum* reduces the lanes to a single scalar by repeatedly adding adjacent pairs in a tree‑like fashion, known as **lane reduction**.

Given an array $a \in \mathbb{Z}^n$ consisting of integer lane values, we want to compute

$$
s = \sum_{i=1}^{n} a_i .
$$

In many implementations this is achieved by loading the vector into a SIMD register and performing pairwise adds until one value remains. The resulting scalar should have the same byte representation as NumPy's exact sum.

## Task

Implement `lane_reduce_sum`:

```python
def lane_reduce_sum(a: np.ndarray) -> np.ndarray:
    ...
```

* `a` is a 1‑D NumPy array of integer type (`np.int32`, `np.int64`, etc.).
* The function must return a **scalar NumPy array** whose dtype matches that of the input.
* Do **not** use Python loops; rely on NumPy’s vectorised operations.

The returned value should be byte‑exactly equal to `np.sum(a)` cast back to the original dtype.

## Example

```python
import numpy as np
a = np.array([1, 2, 3, 4], dtype=np.int32)
s = lane_reduce_sum(a)
print(s)          # scalar array with value 10
print(s.dtype)    # int32
```

## What the gate checks

The grader computes a reference sum using `np.sum`, then compares your output to that reference via `scorer.byte_exact_fraction`.  
Your solution passes only if the byte‑exact fraction is **1.0** (identical bytes). All other properties are irrelevant for this task.
