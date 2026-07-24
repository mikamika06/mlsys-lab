## Context

Modern LLM inference engines (vLLM, TGI) avoid allocating one contiguous KV-cache
buffer per request. Instead the KV-cache is divided into fixed-size **blocks**
(pages) stored in a shared physical backing array. Each request carries a
**block table** that maps its logical block indices to physical block indices in
that backing store.

For a request with $T$ valid tokens and block size $B$, the number of logical
blocks is $\lceil T / B \rceil$. Token at logical position $p$ lives at:

$$
\text{logical block} = \lfloor p / B \rfloor, \qquad
\text{slot in block} = p \bmod B, \qquad
\text{physical block} = \text{block\_table}\!\bigl[\lfloor p / B \rfloor\bigr].
$$

The physical store is a 3-D array of shape $(P, B, D)$ where $P$ is the number
of physical blocks and $D$ is the head (hidden) dimension. Given the block table
and the count of valid tokens, the goal is to reconstruct the contiguous logical
KV tensor of shape $(T, D)$ by gathering entries from the physical store.

No floating-point arithmetic is performed — the task is a pure index-gather
operation — so correctness means reproducing the **exact same bytes** for every
element.

## Task

Implement `reconstruct_logical_kv`:

```python
import numpy as np

def reconstruct_logical_kv(
    physical_store: np.ndarray,   # shape (P, B, D), dtype float32
    block_table: list[int],       # maps logical block -> physical block
    block_size: int,              # B
    num_valid_tokens: int,        # T, may not fill last block
) -> np.ndarray:
    """Return contiguous logical KV of shape (T, D), dtype float32."""
    ...
```

Constraints:

- $1 \le T \le \lvert\text{block\_table}\rvert \times B$.
- Every entry in `block_table` is a valid index into the first axis of
  `physical_store`.
- Use NumPy only; no Python-level loops over tokens in the final solution.

## Example

```python
import numpy as np

physical_store = np.array([
    [[10, 20], [30, 40]],   # physical block 0
    [[50, 60], [70, 80]],   # physical block 1
    [[ 1,  2], [ 3,  4]],   # physical block 2
], dtype=np.float32)

block_table = [2, 0]        # logical 0 → phys 2, logical 1 → phys 0
block_size = 2
num_valid_tokens = 3

# Token 0: block_table[0]=2, slot 0 → [1, 2]
# Token 1: block_table[0]=2, slot 1 → [3, 4]
# Token 2: block_table[1]=0, slot 0 → [10, 20]

result = reconstruct_logical_kv(physical_store, block_table, block_size, num_valid_tokens)
# array([[ 1.,  2.],
#        [ 3.,  4.],
#        [10., 20.]])
```

## What the gate checks

The single gate metric is `byte_exact_fraction`. The grader builds several
deterministic (seeded) physical stores and block tables, computes the reference
logical KV using direct NumPy fancy indexing, then compares the student's output
byte-for-byte. The gate passes only when `byte_exact_fraction == 1.0` across
**every** test case — any off-by-one index, wrong dtype, or shape mismatch
produces a 0 result.
