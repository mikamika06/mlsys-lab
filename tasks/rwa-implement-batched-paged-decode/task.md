## Context

Production attention systems often store key and value tensors in paged memory so that
variable-length sequences can share a fixed-size KV cache. A block table maps logical
token positions to physical pages in the KV pool.

For one decode step, each request has one query vector $q \in \mathbb{R}^{d}$ and
attends over its existing key/value history. The attention output is

$$
o = \sum_{i=1}^{L} \mathrm{softmax}(s)_i v_i ,
$$

where the scores are

$$
s_i = \frac{q^\top k_i}{\sqrt{d}}
$$

and $k_i, v_i$ are gathered from the logical sequence order. The block table is needed
because the physical KV pool is not stored contiguously.

For a batch of requests, request $b$ has a query $q_b$, a sequence length $L_b$, and a
block table that maps logical blocks to physical blocks. If the block size is $B$, the
logical token position $t$ belongs to block $\lfloor t/B \rfloor$ and offset
$t \bmod B$ inside that block.

## Task

Implement `batched_paged_decode`:

```python
def batched_paged_decode(
    q: np.ndarray,
    k_cache: np.ndarray,
    v_cache: np.ndarray,
    block_tables: np.ndarray,
    seq_lens: np.ndarray,
    block_size: int,
) -> np.ndarray:
    ...
```

The inputs are:

- `q`: float array with shape $(batch, d)$ containing one query token per request.
- `k_cache`: float array with shape $(num_blocks, block_size, d)$ containing paged keys.
- `v_cache`: float array with shape $(num_blocks, block_size, d)$ containing paged values.
- `block_tables`: integer array with shape $(batch, max_blocks)$ mapping logical blocks to
  physical blocks.
- `seq_lens`: integer array with shape $(batch,)$ containing each request's valid token count.
- `block_size`: number of tokens per KV block.

Return a float64 array of shape $(batch, d)$ containing the attention output for every
request.

The implementation should gather tokens according to `block_tables` and apply normal
scaled dot-product attention independently for each request.

## Example

```python
import numpy as np

q = np.array([[1.0, 0.0]])
k_cache = np.array([[[1.0, 0.0], [0.0, 1.0]]])
v_cache = np.array([[[10.0, 0.0], [0.0, 20.0]]])
block_tables = np.array([[0]])
seq_lens = np.array([2])

out = batched_paged_decode(
    q, k_cache, v_cache, block_tables, seq_lens, 2
)
# approximately [[8.807970, 11.920300]]
```

## What the gate checks

The gate builds several paged KV batches and computes an independent NumPy oracle by
gathering the logical KV sequence and evaluating the attention equation in float64.
The returned tensor is compared with the oracle using maximum absolute error:

$$
\max_{i,j} |o_{i,j} - \hat{o}_{i,j}|.
$$

The implementation passes when this value is less than $10^{-5}$. A solution that
incorrectly assumes physical blocks are already in logical order will fail on shuffled
block tables.
