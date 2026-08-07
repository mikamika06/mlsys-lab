## Context

PagedAttention-style inference servers (vLLM and similar) do not store each
request's KV cache in its own contiguous buffer. Instead there is one large
**physical pool** of fixed-size blocks (pages), shape
$(\text{num\_physical\_blocks}, \text{block\_size}, d)$, shared by every
request in the batch. Each request owns a **block table**: a small array
mapping its own *logical* block index $b$ to whichever *physical* block
$\text{block\_table}[b]$ currently backs it. Logical position $p$ within a
request therefore lives at

$$
\text{physical block} = \text{block\_table}\!\left[\left\lfloor p / \text{block\_size} \right\rfloor\right], \qquad
\text{slot} = p \bmod \text{block\_size}.
$$

When a batch of ragged requests (each with a different existing context
length) all generate their next token(s) in the same step, the new K/V
vectors for the whole batch arrive **packed** together — one flat
$(\text{total\_new\_tokens}, d)$ array, split per request by
`cu_new_seqlens` (cumulative offsets, same convention as
FlashAttention's varlen API). Each request also carries `seq_start_pos[r]`,
the number of tokens already resident in its cache — i.e. the logical
position of its first newly appended token. Writing this batch into the
pool means resolving, **per new token**, which physical block/slot it
belongs to and storing it there — get this addressing wrong for one request
and you silently corrupt another request's cache (blocks are shared
physical memory).

## Task

Implement `append_paged_kv`:

```python
def append_paged_kv(k_pool, v_pool, new_k, new_v, cu_new_seqlens, seq_start_pos, block_tables, block_size):
    ...
```

* `k_pool`, `v_pool` — `(num_physical_blocks, block_size, d)` float64 arrays,
  the shared physical pool. **Mutate them in place.**
* `new_k`, `new_v` — `(total_new_tokens, d)` float64 arrays: every request's
  new K/V vectors, concatenated back-to-back.
* `cu_new_seqlens` — 1-D int array, length `num_requests + 1`. Request `r`'s
  new tokens are `new_k[cu_new_seqlens[r]:cu_new_seqlens[r+1]]`.
* `seq_start_pos` — 1-D int array, length `num_requests`: tokens already in
  request `r`'s cache before this call.
* `block_tables` — list of 1-D int arrays, length `num_requests`.
  `block_tables[r][b]` is the physical block backing request `r`'s logical
  block `b`.
* `block_size` — int, token slots per physical block.

For request `r`'s `i`-th new token (`0`-indexed within that request), its
absolute logical position is `pos = seq_start_pos[r] + i`. Write
`new_k[cu_new_seqlens[r] + i]` to
`k_pool[block_tables[r][pos // block_size], pos % block_size]`, and the
same for `v_pool` / `new_v`. The function returns `None`; all output is via
the in-place mutation of `k_pool` and `v_pool`.

## Example

```python

block_size = 4
k_pool = [[[0 for _ in range(2)] for _ in range(block_size)] for _ in range(3)]
v_pool = [[[0 for _ in range(2)] for _ in range(block_size)] for _ in range(3)]

# one request, already has 3 tokens cached, now appends 2 more.
# block_table = [0, 1] -> logical block 0 is physical block 0, logical
# block 1 is physical block 1.
new_k = [[1.0, 1.0], [2.0, 2.0]]
new_v = [[9.0, 9.0], [8.0, 8.0]]
cu_new_seqlens = [0, 2]
seq_start_pos = [3]
block_tables = [[0, 1]]

append_paged_kv(k_pool, v_pool, new_k, new_v, cu_new_seqlens, seq_start_pos, block_tables, block_size)
# token at logical pos 3 -> block 3//4=0, slot 3 -> k_pool[0, 3] == [1, 1]
# token at logical pos 4 -> block 4//4=1, slot 0 -> k_pool[1, 0] == [2, 2]
```

## What the gate checks

The grader builds several random ragged batches (3 requests each, random
existing/new lengths, `block_size=4`, distinct randomly-assigned physical
blocks per request) and pre-fills each request's *existing* tokens into the
pool at their correct paged locations.

- **exact_match** — after calling your `append_paged_kv`, the ENTIRE pool
  (`k_pool` and `v_pool`) must be byte-for-byte identical to an independent
  reference that performs the same scatter — this catches wrong block/slot
  arithmetic *and* accidental writes into another request's blocks.
- **max_abs_err** — the grader then reads each request's full context
  (existing + newly appended tokens) back out of your mutated pool via its
  block table, runs causal attention against fresh random queries, and
  compares to attention computed directly over the known ground-truth
  (un-paged) K/V. Must be `<= 1e-5`.

Both gates must pass.
