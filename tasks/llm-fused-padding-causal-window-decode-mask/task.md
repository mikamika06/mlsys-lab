## Context

In transformer models the attention mask determines which tokens a query can attend to. For decoding we need three constraints simultaneously:
- **Padding** – padded positions must never be attended.
- **Causal** – a token may only attend to itself and earlier tokens.
- **Sliding‑window** – for long sequences we restrict each token to look back at most $w$ steps.

The mask is a boolean array of shape $(B, T, T)$ where $B$ is the batch size, $T$ the sequence length. For a target position $i$ and source position $j$, the entry is ``True`` iff all three constraints are satisfied.

## Task

Implement `fused_decode_mask(padding_mask: np.ndarray, window_size: int) -> np.ndarray`:

```python
def fused_decode_mask(padding_mask: np.ndarray, window_size: int) -> np.ndarray:
    ...
```

- `padding_mask` has shape $(B,T)$ and contains ``True`` for real tokens and ``False`` for padding.
- `window_size` is a positive integer $w$.
- The function must return a boolean array of shape $(B,T,T)` that satisfies the three constraints described above.  
  The result should be of type ``np.bool_``.

The implementation must use only NumPy vectorised operations; no Python loops over tokens.

## Example

```python
import numpy as np
pad = np.array([[True, True, False, True]])
mask = fused_decode_mask(pad, window_size=2)
print(mask.astype(int))
# [[[1 0 0 0]
#   [1 1 0 0]
#   [0 0 0 0]
#   [0 0 0 1]]]
```

## What the gate checks

The grader computes a reference mask using the same mathematical definition and compares it to your output with the metric `max_abs_err`. Your solution passes only if the maximum absolute difference between the two masks is at most $10^{-6}$, i.e. the masks are identical.
