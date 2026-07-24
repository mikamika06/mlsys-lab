## Context

Production inference servers pack many variable-length sequences into a
single flat batch of tokens to keep GPUs busy (this is exactly what
`cu_seqlens` means in FlashAttention's *varlen* API, or what vLLM calls a
"packed" / "ragged" prefill batch). If segment $s$ occupies token positions
$[\text{cu\_seqlens}_s,\ \text{cu\_seqlens}_{s+1})$, then for a query at
position $i$ inside segment $s$, attention must only be computed over keys
$j$ such that

$$
j \le i \quad \text{AND} \quad \text{cu\_seqlens}_s \le j < \text{cu\_seqlens}_{s+1}.
$$

A common, subtle bug is to build the mask using only the **global** causal
condition $j \le i$ over the whole packed tensor and forget the segment
boundary. Since the sequences are concatenated back-to-back, this silently
lets every sequence attend into the *tail of the previous sequence* (and any
earlier ones) as if it were part of its own left context — a real
cross-request information leak, not just a numerical rounding issue.

## Task

Fix `ragged_causal_attention`:

```python
def ragged_causal_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, cu_seqlens: np.ndarray) -> np.ndarray:
    ...
```

* `Q, K, V` — `(n, d)` arrays: `num_segments` sequences concatenated along
  the token axis.
* `cu_seqlens` — 1-D integer array of length `num_segments + 1`, e.g.
  `[0, 3, 7, 10]` means segment 0 is tokens `0:3`, segment 1 is tokens
  `3:7`, segment 2 is tokens `7:10`.

Compute scaled dot-product attention with scale $1/\sqrt{d}$, where row $i$
may attend to column $j$ **iff** $j \le i$ **and** $j$ belongs to the same
segment as $i$. Everything else must be masked to $-\infty$ before the
softmax (so allowed rows still normalize to sum to 1 over their own
segment). Return the `(n, d)` attention output.

The provided starter builds a single mask with `col <= row` over the whole
packed batch, ignoring `cu_seqlens` entirely — fix it so each row is also
restricted to its own segment.

## Example

```python
import numpy as np

# two 1-token "sequences" packed together: cu_seqlens = [0, 1, 2]
Q = np.array([[1.0, 0.0], [0.0, 1.0]])
K = np.array([[1.0, 0.0], [0.0, 1.0]])
V = np.array([[10.0, 0.0], [0.0, 20.0]])
cu_seqlens = np.array([0, 1, 2])

out = ragged_causal_attention(Q, K, V, cu_seqlens)
# row 1 (segment 1, token 0) may ONLY see column 1 (its own segment) even
# though column 0 <= row 1 under a naive global causal mask -- so out[1]
# must equal V[1] = [0, 20], not a blend of V[0] and V[1].
```

## What the gate checks

A single gate, **max_abs_err**, compares your output against a reference
that computes the exact same attention in `float64` but derives the
allowed-position mask from `cu_seqlens` (same-segment AND causal). The
grader runs 8 random packed batches (2-4 segments of length 1-5 each,
random feature dimension). The buggy global-mask implementation attends
across segment boundaries and produces errors far above `1e-2` on
boundary-adjacent rows; a correct fix must reach `max_abs_err <= 1e-5` on
every case.
