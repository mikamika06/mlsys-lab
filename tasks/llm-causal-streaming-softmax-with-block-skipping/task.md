## Context

In transformer attention each query attends to a set of keys.  
The raw scores are collected in an $n\times n$ matrix $S$, where row $i$
contains the logits for query $i$ against all $n$ keys.  A *causal* mask
prevents a query from attending to future keys: entries $(i,j)$ with $j>i$
are set to $-\infty$.  After masking, a softmax is applied row‑wise,
producing a probability distribution over the allowed keys.

When the key space is divided into blocks of size $B$, an entire block
may be fully masked for all queries.  In that case the softmax over that
block can be skipped without affecting correctness – this is the *block
skipping* optimisation used in FlashAttention and other efficient
attention kernels.

## Task

Implement a function

```python
def streaming_causal_softmax(logits: np.ndarray,
                             mask: np.ndarray,
                             block_size: int) -> np.ndarray:
```

- `logits` – a 2‑D NumPy array of shape $(n,n)$ containing the raw
  attention scores.
- `mask` – a boolean array of the same shape; `True` indicates that
  the corresponding key is valid, `False` means it should be treated as
  $-\infty$ (probability zero).
- `block_size` – an integer $>0$ representing the size of KV blocks.
  The implementation does **not** need to actually skip blocks for
  correctness; it must simply respect the mask and causal constraint.

The function should return a NumPy array of shape $(n,n)$ containing the
softmax probabilities per row.  Positions that are masked or causally
forbidden must contain zero.  All computations must use `float64`.

## Example

```python
import numpy as np

logits = np.array([[0, 1, -2],
                   [3, 0, 1],
                   [-1, 4, 0]], dtype=np.float64)

mask   = np.array([[True, True, False],
                   [True, True, True ],
                   [False, True, True ]], dtype=bool)

out = streaming_causal_softmax(logits, mask, block_size=2)
print(out)
```

The expected output (rounded to 3 decimals) is

```
[[0.731 0.269 0.000]
 [1.000 0.000 0.000]
 [0.000 0.982 0.018]]
```

## What the gate checks

The grader computes a reference softmax that applies both the causal
mask and the provided boolean mask, using NumPy only.  
It then evaluates the maximum absolute error between the candidate’s
output and this reference with `arena.scorers.max_abs_err`.  
A solution passes if the error is at most $10^{-5}$.

No performance requirement is enforced; a correct but slow implementation
will still succeed.
