## Context

Unstructured pruning zeroes out individual weights in a tensor. A dense
format keeps every element regardless, but a compressed-sparse format stores
only the nonzero values alongside a bitmask indicating which elements survive.

For a tensor of $n$ fp16 elements with $\text{nnz}$ nonzero entries, the dense
footprint is simply

$$\text{dense} = 2\,n \quad \text{bytes}$$

since each fp16 value occupies 2 bytes. A bitmask+values compressed format uses
two components:

- **Bitmask**: one bit per element to flag nonzero status, costing
  $\lceil n / 8 \rceil$ bytes (each byte packs 8 flags).
- **Values**: the surviving fp16 nonzero values, costing $2 \cdot \text{nnz}$
  bytes.

$$\text{sparse} = \left\lceil \frac{n}{8} \right\rceil + 2 \cdot \text{nnz}
\quad \text{bytes}$$

The size ratio $\rho = \text{dense} / \text{sparse}$ quantifies compression
benefit: $\rho > 1$ means the sparse representation is smaller than the
dense original.

## Task

Implement `compressed_sparse_footprint`:

```python
import numpy as np

def compressed_sparse_footprint(tensor: np.ndarray) -> tuple[int, int, float]:
    ...
```

It receives a 1-D NumPy array of `float16` values — a flattened,
unstructured-pruned weight tensor — and must return a 3-tuple:

1. `sparse_bytes` (`int`): the compressed-sparse footprint
   $\lceil n/8 \rceil + 2 \cdot \text{nnz}$.
2. `dense_bytes` (`int`): the dense fp16 footprint, equal to the tensor's
   `.nbytes`.
3. `size_ratio` (`float`): `dense_bytes / sparse_bytes`. If `sparse_bytes`
   is 0 (which cannot happen for $n > 0$), return `float('inf')`.

Do not use Python-level `for` loops over elements — NumPy aggregation
functions like `np.count_nonzero` are expected.

## Example

```python
import numpy as np

t = np.array([1.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             dtype=np.float16)
# n = 10, nnz = 2 (nonzero at indices 0 and 2)

sparse, dense, ratio = compressed_sparse_footprint(t)
# sparse = ceil(10/8) + 2 * 2 = 2 + 4 = 6
# dense  = 10 * 2 = 20
# ratio  = 20 / 6 = 3.333...
```

## What the gate checks

A single gate **`size_ratio`** verifies correctness against a NumPy oracle
that independently recomputes `nnz` via `np.count_nonzero` and
`ceil(n/8)` via integer arithmetic. Six test tensors are checked:

| Case | $n$ | Approx. pruning |
|---|---|---|
| Moderate pruning | 1 000 | 50 % |
| Heavy pruning | 10 000 | 90 % |
| No pruning | 100 | 0 % |
| Total pruning | 500 | 100 % |
| Single element | 1 | — |
| Non-byte-aligned $n$ | 13 | ~23 % |

For each case the oracle asserts:

- `sparse_bytes` $= \lceil n/8 \rceil + 2 \cdot \text{nnz}$
- `dense_bytes` $= $ `tensor.nbytes`
- $\left|\texttt{size\_ratio} - \dfrac{\texttt{dense\_bytes}}{\texttt{sparse\_bytes}}\right| \le 10^{-6}$

All three checks must pass for every case to earn the gate.
