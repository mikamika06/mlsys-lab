## Context

Production inference systems often compress neural network weights before serving them. One common format combines structured sparsity, low-bit values, and small metadata overhead.

For a $2:4$ sparse matrix, every group of four consecutive weights stores exactly two nonzero values. The two selected positions in the group are represented by $2$-bit indices. The two nonzero values are stored as int4 values, requiring $4$ bits each. A scale factor is stored per group using fp16.

For a group containing $g$ values with $k$ nonzero entries, the stored size is the sum of value bytes, index metadata bytes, and scale bytes:

$$
S = k \cdot \frac{4}{8} + k \cdot \frac{2}{8} + 2 .
$$

The dense fp16 representation stores every element using two bytes:

$$
S_{\mathrm{dense}} = 2 \cdot n ,
$$

where $n$ is the number of elements. The served size ratio is

$$
R = \frac{S_{\mathrm{dense}}}{S}.
$$

## Task

Implement `combined_served_size(weight, group_size)`:

```python
def combined_served_size(weight: np.ndarray, group_size: int) -> float:
    ...
```

The input is a NumPy array of weights and a group size. The array is flattened into consecutive groups. Each group contains `group_size` elements and follows the $2:4$ sparse storage rule, meaning exactly half of the elements in each group are nonzero.

Return the compressed representation size in bytes as a Python float. The format stores:

- each nonzero value as one int4 value ($0.5$ bytes),
- each nonzero position as a 2-bit index,
- one fp16 scale value per group ($2$ bytes).

The function should compute the storage required by the actual compressed representation, not the size of the input array.

## Example

```python
import numpy as np

w = np.array([
    1.0, 0.0, 0.0, 2.0,
    0.0, 3.0, 4.0, 0.0
], dtype=np.float32)

size = combined_served_size(w, 4)
# 2 nonzero values per group:
# values: 4 * 0.5 bytes
# indices: 4 * 2 / 8 bytes
# scales: 2 groups * 2 bytes
# total: 7.0 bytes
```

## What the gate checks

The grader builds $2:4$ sparse NumPy weights and computes the reference compressed size using the storage formula from the representation itself. It then compares the served size ratio against the oracle ratio. The returned value must produce a ratio matching the oracle within $10^{-6}$.

A solution that uses incorrect metadata width, ignores scale storage, or assumes values are stored as fp16 fails the gate.
