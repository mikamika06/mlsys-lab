## Context

PagedAttention stores each sequence's KV-cache in fixed-size **physical blocks**
drawn from a shared pool, instead of one contiguous per-sequence buffer. A
**block table** maps a sequence's *logical* block index $0, 1, 2, \dots$ to the
*physical* block id that actually holds those tokens' keys/values in the pool.
This indirection is what lets vLLM-style servers share memory across sequences
and avoid pre-allocating a worst-case-length buffer per request.

Two operations are needed:

1. **Append** — write newly generated tokens' K/V vectors into the cache,
   allocating a fresh physical block from a free list whenever the current
   logical block is full.
2. **Gather + attend** — reconstruct the logical $K, V \in \mathbb{R}^{n\times D}$
   sequence by following the block table into the physical pool, then run
   ordinary single-query scaled dot-product attention:
$$
\text{out} = \operatorname{softmax}\!\left(\frac{q K^\top}{\sqrt{D}}\right) V .
$$

## Task

Implement two functions.

```python
def paged_append(k_pool: list[list[list[float]]], v_pool: list[list[list[float]]], block_table: list[int], free_blocks: list[int], new_k: list[list[float]], new_v: list[list[float]], block_size: int) -> int:
    ...

def gather_and_attend(k_pool, v_pool, block_table, block_size, seq_len, q):
    ...
```

**`paged_append`**

* `k_pool`, `v_pool` — `(num_phys_blocks, block_size, D)` float64 arrays, the
  shared physical pool. Mutate **in place**.
* `block_table` — `list[int]`, physical block id backing each already-allocated
  logical block of *this* sequence (may start empty). Mutate in place: append
  exactly as many new entries as new physical blocks are needed.
* `free_blocks` — `list[int]`, available physical block ids, treated as a
  queue: the next block to allocate is always `free_blocks.pop(0)`. Mutate in
  place.
* `new_k`, `new_v` — `(L, D)` float64 arrays of the new tokens' key/value
  vectors.
* `block_size` — tokens per physical block.

Let `n_before = len(block_table) * block_size` be the number of tokens already
resident. New token `i` (0-indexed) is written to logical position
`n_before + i`. Whenever that position's logical block index
`(n_before + i) // block_size` does not yet exist in `block_table`, allocate a
fresh physical block (`free_blocks.pop(0)`) and append it to `block_table`
*before* writing that token.

Return `n_before + L`, the new total resident length.

**`gather_and_attend`**

* `k_pool`, `v_pool`, `block_table`, `block_size` as above.
* `seq_len` — number of valid logical tokens (may be less than
  `len(block_table) * block_size` if the last block is only partly full).
* `q` — `(D,)` float64 query vector.

Gather the logical `K`, `V` of length `seq_len` by indexing the pool with
`block_table`, then return `softmax(q @ K.T / sqrt(D)) @ V` as a `(D,)` array.

## Example

```python

block_size, D, num_phys = 2, 4, 5
k_pool = [[[0.0 for _ in range(D)] for _ in range(block_size)] for _ in range(num_phys)]
v_pool = [[[0.0 for _ in range(D)] for _ in range(block_size)] for _ in range(num_phys)]
block_table = []
free_blocks = [0, 1, 2, 3, 4]

new_k = [[0.0 for _ in range(D)] for _ in range(3)]
new_v = [[0.0 for _ in range(D)] for _ in range(3)]
seq_len = paged_append(k_pool, v_pool, block_table, free_blocks, new_k, new_v, block_size)
# block_table now has 2 entries (ceil(3/2) blocks used), free_blocks lost 2 ids

q = [0.0 for _ in range(D)]
out = gather_and_attend(k_pool, v_pool, block_table, block_size, seq_len, q)
# out.shape == (D,)
```

## What the gate checks

* **bookkeeping_exact** — after several rounds of `paged_append` (growing the
  sequence in chunks), your `block_table`, `free_blocks`, and the written
  region of `k_pool`/`v_pool` must match a reference implementation exactly
  (allocation order, block-table growth, returned `seq_len`).
* **max_abs_err** — the output of `gather_and_attend` on the resulting cache
  must match a Python attention oracle to within `1e-6` max absolute error.
