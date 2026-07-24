## Context

PagedAttention-style KV caches don't store a request's key/value history in
one contiguous buffer. Instead there is a single shared physical pool of
fixed-size blocks (pages), shape
$(\text{num\_physical\_blocks}, \text{block\_size}, d)$, and each request
owns a small **block table** mapping its own logical block index $b$ to
whichever physical block $\text{block\_table}[b]$ currently backs it (that
mapping can be — and in a real allocator usually is — an arbitrary
shuffled permutation, not the identity). Logical position $p$ therefore
lives at

$$
\text{physical block} = \text{block\_table}\!\left[\left\lfloor p / \text{block\_size} \right\rfloor\right], \qquad
\text{slot} = p \bmod \text{block\_size}.
$$

At decode time, a new query token must attend over every *valid* cached
token — but a request's cached length `seq_len` is rarely an exact multiple
of `block_size`: its last logical block is only partially filled. The
remaining slots in that block (and every physical block *not* referenced by
the block table) may hold another request's live data or stale leftovers
from a freed block. A correct kernel must read **exactly** the `seq_len`
valid positions and nothing else.

## Task

Implement `paged_attention`:

```python
def paged_attention(q, k_pool, v_pool, block_table, seq_len, block_size):
    ...
```

* `q` — `(d,)` query vector for the newly generated token.
* `k_pool`, `v_pool` — `(num_physical_blocks, block_size, d)` float64 arrays:
  the shared physical pool.
* `block_table` — 1-D int array of length `ceil(seq_len / block_size)`;
  `block_table[b]` is the physical block backing this request's logical
  block `b`.
* `seq_len` — number of valid cached tokens for this request.
* `block_size` — token slots per physical block.

Gather the `seq_len` valid K/V vectors by following `block_table` (do **not**
read past `seq_len`, even inside an otherwise-valid block), then compute
standard scaled dot-product attention — no causal mask is needed, every
cached token already precedes the query:

$$
\text{out} = \operatorname{softmax}\!\left(\frac{K q}{\sqrt{d}}\right)^{\!\top} V .
$$

Return the `(d,)` output vector.

## Example

```python
import numpy as np

block_size = 4
d = 2
# seq_len=5 needs 2 logical blocks; block_table is a SHUFFLED mapping.
block_table = np.array([2, 0])   # logical block 0 -> physical block 2, etc.
k_pool = np.random.randn(4, block_size, d)   # 4 physical blocks total
v_pool = np.random.randn(4, block_size, d)

# place the real 5-token K/V sequence into its physical slots...
# (positions 0-3 -> physical block 2, position 4 -> physical block 0, slot 0)

out = paged_attention(q, k_pool, v_pool, block_table, seq_len=5, block_size=block_size)
# out has shape (2,)
```

## What the gate checks

A single gate, **max_abs_err**, compares your output against
`softmax(K_seq @ q / sqrt(d)) @ V_seq` computed directly (in float64) on the
known ground-truth logically-contiguous `K_seq`, `V_seq` the grader used to
populate the pool. Across 10 random cases (random `d`, random `seq_len` from
1 to `3 * block_size`, a randomly SHUFFLED `block_table`, and every unused
slot / unreferenced block filled with high-magnitude adversarial "poison"
values) your reconstructed output must match to `<= 1e-5`. Following the
block table incorrectly, assuming an identity mapping, or reading past
`seq_len` into the poisoned padding all pull in the wrong K/V vectors and
blow the softmax up far past this tolerance.
