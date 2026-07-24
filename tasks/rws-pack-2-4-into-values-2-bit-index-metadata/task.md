## Context

In many sparse‑matrix libraries a *2:4* compression format keeps the two largest magnitude entries in every block of four consecutive elements. The remaining two positions are omitted from storage. To reconstruct the original layout we need not only the retained values but also their original indices within each 4‑element block. Since an index can take one of four values, it fits into exactly two bits.

The compressed representation therefore consists of

- a dense array `values` holding the selected magnitudes,
- a metadata array `indices` where each entry is a 2‑bit integer in $\{0,1,2,3\}$ indicating the position of the corresponding value inside its block.

Both arrays are stored contiguously; for grading we concatenate them into a single byte buffer.

## Task

Implement the function

```python
def pack_2_of_4(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ...
```

`a` is a one‑dimensional NumPy array of length divisible by four. The function must return two arrays:

1. `values`: a float64 array containing the two largest magnitude elements from each block of four, in the order they appear when scanning the input left to right.
2. `indices`: a uint8 array where each entry is an integer $i\in\{0,1,2,3\}$ giving the original position of the corresponding value within its 4‑element block.

The implementation must be fully vectorised; no explicit Python loops over elements are allowed.

## Example

```python
import numpy as np
a = np.array([0.5, -1.2, 0.0, 3.4,
              2.1, 0.0, -0.7, 0.9], dtype=np.float64)

values, indices = pack_2_of_4(a)
print(values)   # [ 3.4  2.1]
print(indices)  # [3 0]
```

The first block `[0.5, -1.2, 0.0, 3.4]` keeps `3.4` (index 3) and `-1.2` (index 1).  
The second block `[2.1, 0.0, -0.7, 0.9]` keeps `2.1` (index 0) and `0.9` (index 3).

## What the gate checks

The grader concatenates the two returned arrays into a single byte buffer
```
buffer = np.concatenate([values.astype(np.float64),
                         indices.astype(np.uint8)]).tobytes()
```
and compares this buffer with one produced by an oracle implementation.
The metric `byte_exact_fraction` must equal 1.0, i.e. the two buffers are identical bit‑for‑bit.
