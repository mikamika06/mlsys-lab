## Context

PagedAttention stores every sequence's KV cache in fixed-size **physical
blocks** carved out of one shared pool, instead of one contiguous
per-sequence buffer. A sequence's logical positions are mapped to
physical rows through a **block table**: logical block `b` lives in
physical block `block_table[b]`, and position `pos` within that logical
block sits at offset `pos % block_size` inside it. So the physical row
("slot") holding position `pos` is:

$$
\text{slot}(pos) = \text{block\_table}\!\left[\left\lfloor \tfrac{pos}{\text{block\_size}} \right\rfloor\right] \cdot \text{block\_size} \;+\; (pos \bmod \text{block\_size})
$$

Physical block ids are **not** required to equal their logical index, or
even to be contiguous — the whole point of paging is that a sequence's
blocks can be scattered anywhere in the shared pool. Appending `T` new
tokens means writing each one into its slot via this mapping (the
**slot mapping**, `[slot(existing_len), ..., slot(existing_len+T-1)]`);
those `T` slots may land in more than one physical block if the append
crosses a `block_size` boundary. Reading the sequence back out
(e.g. to attend) means gathering through the very same mapping.

## Task

Implement `paged_append_and_attend`:

```python
def paged_append_and_attend(
    kv_pool_k: np.ndarray, kv_pool_v: np.ndarray,
    block_table: list[int], block_size: int, existing_len: int,
    new_k: np.ndarray, new_v: np.ndarray, q: np.ndarray,
) -> np.ndarray:
    ...
```

- `kv_pool_k`, `kv_pool_v`: `(num_physical_blocks * block_size, d)`, the
  shared physical pool. Positions `0 .. existing_len-1` of THIS sequence
  are already written at their correct slots; every other row may hold
  unrelated data from other sequences (do not assume it is zero or
  meaningful).
- `block_table`: this sequence's logical-block-to-physical-block-id list
  — long enough to cover `existing_len + T` positions. Ids are not
  necessarily contiguous or in logical order.
- `block_size`, `existing_len`: as above.
- `new_k`, `new_v`: `(T, d)`, to be written at positions
  `existing_len .. existing_len+T-1`.
- `q`: `(d,)`, attended after the append, over all `existing_len + T`
  tokens.

Write `new_k`/`new_v` into the pool via `slot(pos)`, gather the full
`existing_len + T` sequence back out via the same mapping (never assume
positions are contiguous in the pool), and return the `(d,)` scaled
dot-product attention output of `q` over the gathered `K`/`V`.

## Example

```python
import numpy as np

block_size = 4
block_table = [2, 0]          # logical block 0 -> physical 2, logical block 1 -> physical 0
existing_len = 3              # 3 tokens already written (all in logical block 0)
new_k = np.random.default_rng(0).standard_normal((3, 8))   # spans blocks 0 and 1
new_v = np.random.default_rng(1).standard_normal((3, 8))
q = np.random.default_rng(2).standard_normal(8)

# pool must already hold the 3 existing tokens at slot(0), slot(1), slot(2)
out = paged_append_and_attend(kv_pool_k, kv_pool_v, block_table, block_size,
                               existing_len, new_k, new_v, q)
# out.shape == (8,)
```

## What the gate checks

The grader builds several seeded scenarios where `existing_len` starts
inside the first logical block and `T` is chosen so the append always
spans at least two logical (and, via a randomly permuted block table,
non-contiguous physical) blocks, with the rest of the pool filled with
unrelated large-magnitude noise. It compares your output to a reference
that reconstructs the same sequence directly by concatenating the
original contiguous `existing_k/v` and `new_k/v` arrays used to build the
pool, and attends in NumPy — never calling your function, never
hardcoding an expected value, and structurally independent of any
slot-mapping bug.

`max_abs_err` is the worst per-case max-abs-error and must be `<= 1e-5`.
Treating the pool as contiguous by absolute position (ignoring
`block_table`), only writing the new tokens without also gathering the
pre-existing ones through the mapping, or mis-computing the block/offset
split all pull in unrelated noise rows and produce a large error.
