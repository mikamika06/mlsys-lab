## Context

In transformer‑style models, attention is computed over a sequence of tokens. When several short sequences are concatenated into one long “packed” sequence, we must prevent tokens from different original sequences from attending to each other. A convenient way is to use a *block‑diagonal* mask: the matrix has ones on the diagonal blocks corresponding to each original sequence and zeros elsewhere.

Let $L = (l_1,\dots,l_k)$ be the list of lengths of the packed subsequences, with total length $N=\sum_i l_i$. The desired mask $M\in \{0,1\}^{N\times N}$ satisfies

$$
M_{pq} =
\begin{cases}
1 & \text{if } p,q \text{ belong to the same original sequence},\\
0 & \text{otherwise}.
\end{cases}
$$

This mask can be used directly in attention weight computations, e.g. by multiplying it with the softmaxed logits.

## Task

Implement `packed_block_diagonal_mask(seq_lengths)`:

```python
def packed_block_diagonal_mask(seq_lengths: Sequence[int]) -> np.ndarray:
    ...
```

It receives a sequence of positive integers describing the lengths of each original subsequence and returns an $N\times N$ boolean NumPy array where entries are `True` inside each block and `False` elsewhere. The implementation must use vectorized NumPy operations only; no explicit Python loops over tokens.

## Example

```python
import numpy as np
lens = [3, 2]
mask = packed_block_diagonal_mask(lens)
print(mask.astype(int))
# [[1 1 1 0 0]
#  [1 1 1 0 0]
#  [1 1 1 0 0]
#  [0 0 0 1 1]
#  [0 0 0 1 1]]
```

## What the gate checks

The grader computes a reference mask using NumPy and compares it byte‑wise to your output with the scorer `byte_exact_fraction`. The candidate must achieve a fraction of exactly $1.0$, meaning every bit matches the oracle’s result.

Additionally, the solution should be robust: it must handle any positive integer lengths list, including single‑element lists or empty input (which yields an empty mask).
