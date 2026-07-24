## Context

Gathering is the operation of extracting a sub‑array according to an index array.  
In NumPy, the idiomatic way to perform a gather is through `np.take` or advanced indexing:

$$\text{gather}(A,i) = A[i] \quad\text{with}\; i \in \mathbb{N}^k.$$

When implemented with Python loops it becomes a series of scalar loads and
writes, which defeats cache locality and incurs a large number of CPU events.
Vectorised gathering using NumPy’s built‑in kernels keeps the work on the
backend library where it can be executed efficiently and in parallel.

## Task

Implement the function `gather`:

```python
def gather(arr: np.ndarray, indices: np.ndarray) -> np.ndarray:
    ...
```

The function receives a 1‑D array `arr` of arbitrary dtype and an integer array
`indices`. It must return a new array containing the elements selected from
`arr` in the order given by `indices`.  
Your implementation **must** use NumPy’s vectorised gather (`np.take`
or advanced indexing) and **must not** contain any Python loop.

## Example

```python
import numpy as np
a = np.array([10, 20, 30, 40])
idx = np.array([3, 1, 0])
y = gather(a, idx)
print(y)   # [40 20 10]
```

## What the gate checks

* **exact_match** – The returned array must be byte‑identical to
`np.take(arr, indices)` for all test cases.
* **line_events** – Using `sys.settrace`, we count the number of line events that occur during a single call to your function.  This forces a truly vectorised implementation: Python loops typically trigger hundreds of line events.

Both metrics must satisfy the thresholds defined in `meta.json`. The reference
implementation below uses only one NumPy call and therefore should pass both
metrics easily.
