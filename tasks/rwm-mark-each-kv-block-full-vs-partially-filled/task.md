## Context

In many language models, key‑value (KV) caches are stored in fixed‑size blocks. When a sequence of length \(L\) is processed with block size \(b\), the cache must allocate \(\lceil L/b\rceil\) blocks. All but possibly the last block contain exactly \(b\) entries and are therefore *full*. The final block may hold fewer than \(b\) entries, making it *partially filled*. Correctly identifying which blocks are full versus partially filled is important for memory‑efficiency analyses.

## Task

Implement `mark_kv_blocks(seq_lengths, block_size)` that returns a one‑dimensional NumPy boolean array. Each element corresponds to a KV block allocated across all sequences in the order they appear. The value should be `True` if the block is full and `False` otherwise.

The function signature must be:

```python
def mark_kv_blocks(seq_lengths: np.ndarray, block_size: int) -> np.ndarray:
    ...
```

- `seq_lengths` – a 1‑D array of non‑negative integers giving the token count for each sequence.
- `block_size` – a positive integer.

The output must be a NumPy array of dtype `bool`. No Python loops over individual blocks are required, but they are allowed if you wish; performance is not graded here.

## Example

```python
import numpy as np
seq_lengths = np.array([5, 12])
block_size   = 4
labels = mark_kv_blocks(seq_lengths, block_size)
print(labels)          # [ True False  True  True  True]
```

Explanation:  
Sequence 1 (length 5) needs two blocks: the first is full (`True`), the second holds one token (`False`).  
Sequence 2 (length 12) needs three full blocks.

## What the gate checks

The grader computes a reference array using NumPy and compares it to your output with `np.array_equal`. The metric `exact_match` must be `1.0`.
