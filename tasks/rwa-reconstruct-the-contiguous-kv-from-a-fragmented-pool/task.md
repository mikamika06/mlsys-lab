## Context

A paged KV pool stores many sequences' key/value rows interleaved in one
shared physical buffer, `num_physical_blocks * block_size` rows deep. A
single sequence's logical positions `0, 1, 2, \dots` are scattered across
that pool according to its **block table**: logical block $b$'s tokens
live in physical block `block_table[b]`, at

$$
\text{slot}(pos) = \text{block\_table}\!\left[\left\lfloor \tfrac{pos}{\text{block\_size}} \right\rfloor\right] \cdot \text{block\_size} \;+\; (pos \bmod \text{block\_size})
$$

Physical block ids are assigned independently of logical order — two
adjacent logical blocks can land in physical blocks that are far apart
(or even in reverse order), and every physical row *not* named by this
sequence's block table belongs to some other sequence (or is unused) and
must not leak into the reconstruction.

## Task

Implement `reconstruct_contiguous_kv`:

```python
def reconstruct_contiguous_kv(kv_pool: list[list[float]], block_table: list[int], block_size: int, seq_len: int) -> list[list[float]]:
    ...
```

- `kv_pool`: `(num_physical_blocks * block_size, d)`, the shared pool.
- `block_table`: this sequence's logical-block-to-physical-block-id
  list, `len(block_table) * block_size >= seq_len`.
- `block_size`, `seq_len`: as above.
- Return a `(seq_len, d)` array whose row `pos` is
  `kv_pool[slot(pos)]` for every `pos` in `range(seq_len)`, using the
  slot formula above.

## Example

```python

block_size = 4
block_table = [3, 0, 5]   # logical 0,1,2 -> physical 3,0,5 (out of order, non-contiguous)
seq_len = 10              # spans all 3 logical blocks (last one partially)

kv = reconstruct_contiguous_kv(kv_pool, block_table, block_size, seq_len)
# kv.shape == (10, kv_pool.shape[1])
# kv[0]  == kv_pool[3*4 + 0]   (logical block 0, offset 0 -> physical block 3)
# kv[4]  == kv_pool[0*4 + 0]   (logical block 1, offset 0 -> physical block 0)
# kv[9]  == kv_pool[5*4 + 1]   (logical block 2, offset 1 -> physical block 5)
```

## What the gate checks

The grader draws a seeded contiguous tensor, scatters it into a
fragmented pool using a random permutation for the block table (with
extra unrelated large-magnitude noise filling every other row of the
pool), and passes only the pool, block table, `block_size`, and
`seq_len` to your function — it independently keeps the original
contiguous tensor to compare against, so the oracle never calls your
function or hardcodes an expected value.

`max_abs_err` is the worst per-case max-abs-error between your
reconstruction and the original tensor, and must be `<= 1e-6`. Assuming
physical blocks are visited in logical order, treating the pool as
already contiguous, or mis-computing the block/offset split for `pos`
all pull in rows belonging to other sequences (or the wrong offset) and
produce a large error.
