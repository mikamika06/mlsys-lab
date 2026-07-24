## Context

Inference servers prefill many requests of different prompt lengths at once
by packing all their tokens contiguously into one flat batch instead of
padding every sequence out to the longest one — this is what FlashAttention's
*varlen* API and vLLM's "packed prefill" call `cu_seqlens`. Given
`num_segments` sequences with lengths $\ell_0, \ell_1, \dots$, the cumulative
sequence-length array is

$$
\text{cu\_seqlens} = \Big[0,\ \ell_0,\ \ell_0+\ell_1,\ \dots,\ \textstyle\sum_s \ell_s\Big],
$$

so segment $s$ occupies token positions $[\text{cu\_seqlens}_s,\
\text{cu\_seqlens}_{s+1})$ of the packed tensors. Every query token still
needs ordinary **causal** attention *restricted to its own segment*: a query
at packed position $i$ inside segment $s$ may attend to key position $j$ iff

$$
j \le i \quad \text{AND} \quad \text{cu\_seqlens}_s \le j < \text{cu\_seqlens}_{s+1}.
$$

Crucially, sequences must never see each other. Even though sequence 2's
tokens sit immediately after sequence 1's in memory, a naive global causal
mask ($j \le i$ over the whole packed tensor, ignoring segment boundaries)
would let sequence 2 attend into the tail of sequence 1 — a real
cross-request information leak, not just a numerical detail.

## Task

Implement `ragged_batched_prefill_attention(Q, K, V, cu_seqlens)`:

```python
def ragged_batched_prefill_attention(Q, K, V, cu_seqlens):
    ...
```

Inputs:
- `Q`, `K`, `V`: NumPy arrays of shape $(n_{tok}, H, d)$ — all segments'
  tokens packed contiguously along axis 0, with $H$ attention heads and
  head dimension $d$ (this mirrors FlashAttention-varlen's
  `(total_tokens, num_heads, head_dim)` layout).
- `cu_seqlens`: 1-D integer array of length `num_segments + 1`, e.g.
  `[0, 3, 7, 10]` means segment 0 is tokens `0:3`, segment 1 is tokens
  `3:7`, segment 2 is tokens `7:10`.

For every segment, independently:
1. Slice out that segment's $Q, K, V$ rows.
2. Compute per-head scaled dot-product attention with scale $1/\sqrt{d}$,
   masked causally **within the segment only** (position $i$ inside the
   segment may attend to positions $0 \dots i$ of the *same* segment,
   nothing from any other segment).
3. Write the segment's output back into the corresponding rows of the
   result.

Return a `float64` NumPy array of shape $(n_{tok}, H, d)$. Tokens from
different segments must never attend to each other, regardless of how close
together they sit in the packed tensor.

## Example

```python
import numpy as np

# two 1-token "sequences" packed together: cu_seqlens = [0, 1, 2]
Q = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])   # shape (2, 1, 2): n_tok=2, H=1, d=2
K = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])
V = np.array([[[10.0, 0.0]], [[0.0, 20.0]]])
cu_seqlens = np.array([0, 1, 2])

out = ragged_batched_prefill_attention(Q, K, V, cu_seqlens)
# token 1 (segment 1's only token) may ONLY see key 1 (its own segment),
# even though key 0 <= query 1 under a naive global causal mask -- so
# out[1] must equal V[1] = [0, 20], not a blend of V[0] and V[1].
```

## What the gate checks

The gate builds several random packed batches with `np.random.default_rng(0)`
(2-5 segments per batch, random lengths, random head counts and head
dimensions) and an oracle that **loops per segment**, slicing `Q`/`K`/`V` by
`cu_seqlens` and running ordinary causal attention in `float64` on each
slice independently, then reassembles the packed output. Your function's
output is compared against this oracle with `max_abs_err`, threshold
$10^{-5}$. Using a single global causal mask over the whole packed tensor
(ignoring segment boundaries) lets later segments leak-attend into earlier
ones and produces large errors on every row except the very first token of
each segment.
