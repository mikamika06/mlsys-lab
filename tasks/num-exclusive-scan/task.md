## Context

The *prefix sum* (also called a scan) of an array $a = (a_0, a_1, \dots , a_{n-1})$ is the sequence
$$s_i = \sum_{j=0}^{i} a_j,$$
for $i = 0,\dots,n-1$.  
An **exclusive** scan shifts this definition one step to the left:
$$e_i = \sum_{j=0}^{i-1} a_j,$$
with the convention that $e_0 = 0$.

Exclusive scans are fundamental in parallel algorithms and many numerical libraries expose them as a primitive. In Python, NumPy provides `np.cumsum` for inclusive scans; an exclusive scan can be built from it by shifting the result.

## Task

Implement the function `exclusive_scan(arr)`:

```python
def exclusive_scan(arr: np.ndarray) -> np.ndarray:
    ...
```

* `arr` is a one‑dimensional NumPy array of any numeric dtype.
* The return value must be a NumPy array of the same shape and dtype as `arr`.
* For each index $i$, the output should contain $\sum_{j=0}^{i-1} arr[j]$; in particular, the first element is zero.

The implementation must use only NumPy operations (no Python loops).

## Example

```python
import numpy as np
a = np.array([3, 1, 4, 1, 5])
e = exclusive_scan(a)
print(e)          # [0 3 4 8 9]
```

## What the gate checks

The grader computes a reference result using NumPy’s `cumsum` and compares it to your output with an exact match. The metric is named **exact_match**; the solution passes only if all test cases produce identical arrays.
