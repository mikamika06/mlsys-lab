## Context

To keep GPUs busy, LLM pretraining almost never pads short documents up to
a fixed sequence length — it **packs** several documents back-to-back into
one training sequence and resets both the position ids and the attention
mask at each document boundary. If document $s$ occupies token positions
given by `segment_ids == s`, then a query at position $i$ (in document $s$)
must only attend to keys $j$ with

$$
j \le i \quad \text{AND} \quad \text{segment\_ids}[j] = \text{segment\_ids}[i].
$$

A common, easy-to-miss bug is building the mask from the **global** causal
condition $j \le i$ alone and forgetting to also reset it at document
boundaries. Because the documents sit back-to-back in memory, this silently
lets every document attend into the tail of the *previous* document — the
model effectively trains on cross-document "context" that never existed in
the source data.

## Task

Fix `packed_attention_with_reset_mask`:

```python
def packed_attention_with_reset_mask(Q: np.ndarray, K: np.ndarray, V: np.ndarray, segment_ids: np.ndarray) -> np.ndarray:
    ...
```

* `Q, K, V` — `(n, d)` arrays: several documents packed along the token
  axis.
* `segment_ids` — `(n,)` int array; `segment_ids[i]` is the document index
  token `i` belongs to (non-decreasing, e.g. `[0,0,0,1,1,2,2,2,2]` for three
  packed documents).

Compute scaled dot-product attention (scale $1/\sqrt{d}$) where row $i$ may
attend to column $j$ **iff** $j \le i$ **and** `segment_ids[j] ==
segment_ids[i]`. Everything else must be masked to $-\infty$ before the
softmax. Return the `(n, d)` output.

The provided starter builds a single mask with `col <= row` over the whole
packed sequence, ignoring `segment_ids` entirely — fix it so the mask also
resets at every document boundary.

## Example

```python
import numpy as np

# two 1-token "documents" packed together
Q = np.array([[1.0, 0.0], [0.0, 1.0]])
K = np.array([[1.0, 0.0], [0.0, 1.0]])
V = np.array([[10.0, 0.0], [0.0, 20.0]])
segment_ids = np.array([0, 1])

out = packed_attention_with_reset_mask(Q, K, V, segment_ids)
# token 1 (document 1) may ONLY see column 1 (its own document), even
# though column 0 <= row 1 under a naive global causal mask -- so out[1]
# must equal V[1] = [0, 20], not a blend of V[0] and V[1].
```

## What the gate checks

A single gate, **max_abs_err**, compares your output against a reference
that derives the allowed-position mask from `segment_ids` (same-document
AND causal), computed in `float64`. The grader runs 8 random packed
batches (2-4 documents of length 2-6 each, random feature dimension) and
plants an ADVERSARIAL twist in every one: the last token of every
non-final document is given a huge, distinctive value vector. This makes
any cross-boundary attention weight — however small — blow up the output
of the following document's early tokens, so leaking mask logic is caught
immediately. A correct fix must reach `max_abs_err <= 1e-5` on every case.
