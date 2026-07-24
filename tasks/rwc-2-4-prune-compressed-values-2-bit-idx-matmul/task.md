## Context

NVIDIA's Ampere+ sparse tensor cores accelerate matmuls under **2:4
structured sparsity**: every group of 4 consecutive elements along a row
must have exactly 2 zeros. The hardware never stores the zeros — it stores
only the **2 surviving values per group** plus a **2-bit index** per
surviving value recording which of the 4 positions it came from. At matmul
time the hardware reconstructs (gathers) the dense operand from this
compressed form on the fly.

For a weight row split into groups of 4, $[w_0, w_1, w_2, w_3]$, pruning
keeps the 2 entries with the largest $|w_i|$ (ties broken toward the lower
index) and zeros the rest:

$$
\text{mask}_j = \begin{cases} 1 & j \in \operatorname{argtop2}_i |w_i| \\ 0 & \text{otherwise} \end{cases}
$$

## Task

Implement `prune24_compress_and_matmul`:

```python
def prune24_compress_and_matmul(W: np.ndarray, X: np.ndarray):
    ...
```

* `W` — `(m, n)` weight matrix, `n` divisible by 4.
* `X` — `(n, p)` input/activation matrix.

For every row of `W`, split its columns into consecutive groups of 4 and
keep the 2 largest-magnitude values per group (ties broken toward the
lower index within the group). Build the compressed representation:

* `values` — `(m, n//2)` float64: the 2 kept values per group, in their
  original left-to-right order.
* `indices` — `(m, n//2)` uint8: each kept value's position (`0`-`3`)
  within its group of 4, same order as `values`.

**Reconstruct** the pruned `W` from `(values, indices)` (scatter each
value back to `group_start + index`) and compute the matmul against `X`.
Return `(mask, values, indices, output)`:

* `mask` — `(m, n)` 0/1 array, 1 where a value survived pruning.
* `values`, `indices` — as above.
* `output` — `(m, p)`, the reconstructed-pruned-`W` times `X`.

## Example

```python
import numpy as np

W = np.array([[4.0, -1.0, 0.5, 3.0]])   # one group of 4
X = np.array([[1.0], [1.0], [1.0], [1.0]])

mask, values, indices, output = prune24_compress_and_matmul(W, X)
# |values| = [4, 1, 0.5, 3] -> top 2 are 4.0 (idx 0) and 3.0 (idx 3)
# mask    = [[1, 0, 0, 1]]
# values  = [[4.0, 3.0]], indices = [[0, 3]]
# pruned row = [4.0, 0, 0, 3.0] -> output = [[7.0]]
```

## What the gate checks

- **mask_exact** — your `mask` must exactly match a from-scratch reference
  (top-2-by-magnitude per group of 4, ties toward the lower index) across
  5 random `(W, X)` pairs of varying shape.
- **max_abs_err** — your `output`, reconstructed purely from your own
  `(values, indices)`, must match `(reference-pruned W) @ X` to `<= 1e-9`
  on the same 5 cases.

Both gates must pass.
