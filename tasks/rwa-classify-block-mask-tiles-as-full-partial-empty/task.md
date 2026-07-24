## Context

Modern attention kernels (e.g. FlashAttention, FlexAttention) operate on **tiles**: rectangular blocks of the full $Q \times K$ attention matrix. Before computing a tile, the runtime classifies it:

- **Full** — every element in the tile is attended (all `True`): the entire block contributes, no element-wise masking needed.
- **Empty** — no element in the tile is attended (all `False`): the block can be completely skipped.
- **Partial** — some elements are `True` and some are `False`: a fine-grained mask must be applied.

Given a block size $B_q$ (rows per query tile) and $B_{kv}$ (columns per key/value tile), the $(i, j)$-th tile spans query indices $[i \cdot B_q,\ (i+1) \cdot B_q)$ and key indices $[j \cdot B_{kv},\ (j+1) \cdot B_{kv})$.

A **mask modulator** `mask_mod(b, h, q_idx, kv_idx)` returns `True` if position $(q\_idx, kv\_idx)$ is attended in batch $b$, head $h$.

The classification of tile $(i, j)$ for a single batch / head is:

$$\text{label}(i,j) = \begin{cases} \texttt{"full"} & \text{if } \forall\, q \in [i B_q, (i+1)B_q),\, k \in [j B_{kv}, (j+1)B_{kv}):\; \text{mask\_mod}(0,0,q,k)=\text{True} \\ \texttt{"empty"} & \text{if } \forall\, q,k \text{ in block}: \text{mask\_mod}(0,0,q,k)=\text{False} \\ \texttt{"partial"} & \text{otherwise} \end{cases}$$

This classification is exactly what `create_block_mask` in FlexAttention computes internally.

## Task

Implement `classify_block_mask_tiles(mask_mod, seq_len_q, seq_len_kv, block_q, block_kv)`:

```python
def classify_block_mask_tiles(mask_mod, seq_len_q, seq_len_kv, block_q, block_kv):
    ...
```

- `mask_mod(b, h, q_idx, kv_idx) -> bool` — mask modulator (use `b=0, h=0`).
- `seq_len_q`, `seq_len_kv` — total sequence lengths.
- `block_q`, `block_kv` — tile sizes in tokens.
- Returns a 2-D list (or NumPy array) of strings of shape `(num_q_blocks, num_kv_blocks)` where `num_q_blocks = ceil(seq_len_q / block_q)` and `num_kv_blocks = ceil(seq_len_kv / block_kv)`. Each entry is `"full"`, `"empty"`, or `"partial"`.

Sequence lengths are guaranteed to be exact multiples of the respective block sizes.

## Example

```python
import math

def causal(b, h, q, k):
    return k <= q

result = classify_block_mask_tiles(causal, 4, 4, 2, 2)
# result[0][0]: q in [0,1], k in [0,1] -> all True  -> "full"
# result[0][1]: q in [0,1], k in [2,3] -> all False -> "empty"
# result[1][0]: q in [2,3], k in [0,1] -> all True  -> "full"
# result[1][1]: q in [2,3], k in [2,3] -> mixed     -> "partial"
# => [["full","empty"],["full","partial"]]
```

## What the gate checks

`exact_match`: The grader exhaustively evaluates `mask_mod(0, 0, q, k)` for every element in every tile, labels each tile ground-truth, and checks that the student's returned grid matches exactly. Score is 1.0 only if all tiles are correctly classified across multiple mask types (causal, sliding window, constant-True, constant-False).
