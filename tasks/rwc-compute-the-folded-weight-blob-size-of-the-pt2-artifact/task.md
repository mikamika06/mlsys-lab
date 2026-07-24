## Context

In a PyTorch model compiled to the `.pt2` format, every learnable parameter and constant is stored in a binary blob.  
The size of this *weight‑blob* is simply the sum over all tensors of their number of elements multiplied by the byte width of their storage dtype:

$$\text{size} \;=\;\sum_{t \in \mathcal{T}} \bigl(\operatorname{numel}(t)\bigr) \times \operatorname{bytes}(t).$$

Here $\mathcal{T}$ denotes the set of all tensors that belong to the artifact, `numel(t)` is the total number of scalar entries in tensor $t$, and `bytes(t)` is the size in bytes of a single element (e.g. 4 for `float32`, 8 for `int64`).  
The goal of this task is to compute that sum given a mapping from names to NumPy arrays.

## Task

Implement the function:

```python
def compute_folded_weight_blob_size(params: dict[str, np.ndarray]) -> int:
    ...
```

`params` maps parameter/constant names to their corresponding `numpy.ndarray`.  
The function must return an integer representing the total number of bytes that would be stored in a compiled `.pt2` weight‑blob.  Use only NumPy operations; do not rely on any external libraries.

## Example

```python
import numpy as np

params = {
    "w1": np.ones((3, 4), dtype=np.float32),   # 12 elements × 4 bytes = 48
    "b1": np.zeros(5, dtype=np.int64),         # 5 elements  × 8 bytes = 40
    "c1": np.arange(6, dtype=np.uint16)        # 6 elements  × 2 bytes = 12
}

size = compute_folded_weight_blob_size(params)
print(size)   # 100
```

The returned value `100` equals $48 + 40 + 12$.

## What the gate checks

The grader verifies that the function returns **exactly** the integer byte count computed by the reference implementation.  No floating‑point tolerance is required; a mismatch causes the submission to fail.
