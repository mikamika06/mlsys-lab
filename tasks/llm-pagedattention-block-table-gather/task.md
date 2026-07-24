## Context

In autoregressive language models each token added to the sequence contributes a key/value pair that is stored in a cache. During decoding the cache is often split into fixed‑size blocks so that only the needed blocks are kept in fast memory and can be paged out when not used. A *block table* maps a logical position \(i\) (the index of the token in the sequence) to a physical location \((b, o)\), where \(b\) is the block identifier and \(o < B\) is the offset inside that block with block size \(B\).

The gather operation must collect the keys and values for an arbitrary set of logical positions from possibly many blocks. The result should be two NumPy arrays \(\mathbf{K}\in\mathbb R^{n\times d}\) and \(\mathbf{V}\in\mathbb R^{n\times d}\), where \(n\) is the number of requested indices and \(d\) is the hidden dimension.

## Task

Implement a function with the following signature:

```python
def paged_gather(blocks: List[Tuple[np.ndarray, np.ndarray]],
                 block_table: Dict[int, Tuple[int,int]],
                 indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ...
```

* `blocks` – list of tuples \((K_{\text{block}}, V_{\text{block}})\).  
  Each element has shape \((B,d)\), where \(B\) is the block size.
* `block_table` – mapping from logical index to a pair \((b,o)\).
* `indices` – NumPy array of logical indices that should be gathered.

The function must return two arrays `(keys, values)` each of shape `(len(indices), d)` and dtype `float64`. The implementation may use any NumPy operations; no external libraries are required.

## Example

```python
import numpy as np

# Two blocks, block size 4, hidden dimension 3
k0 = np.arange(12).reshape(4,3)          # keys of block 0
v0 = np.arange(12,24).reshape(4,3)       # values of block 0
k1 = np.arange(24,36).reshape(4,3)       # keys of block 1
v1 = np.arange(36,48).reshape(4,3)       # values of block 1

blocks = [(k0, v0), (k1, v1)]
block_table = {i: (i//4, i%4) for i in range(8)}   # logical index → (block, offset)

indices = np.array([2, 5, 7])            # gather positions 2, 5 and 7

keys, values = paged_gather(blocks, block_table, indices)
print(keys)
# [[ 2  3  4]
#  [18 19 20]
#  [30 31 32]]

print(values)
# [[12 13 14]
#  [24 25 26]
#  [36 37 38]]
```

## What the gate checks

The grader builds a contiguous reference cache by concatenating all blocks and then indexes it with `indices`. It compares your output against this reference using the scorer `max_abs_err`, which returns the maximum absolute difference between two arrays. Your solution must achieve

\[
\mathrm{max\_abs\_err} \le 10^{-5}.
\]

The gate also verifies that the function exists and can be called with the expected arguments.
