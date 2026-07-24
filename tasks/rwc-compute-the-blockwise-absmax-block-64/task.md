## Context

In block quantization a weight vector $w \in \mathbb{R}^n$ is partitioned into
non-overlapping blocks of size $B$. Each block is normalized independently
using its own scale factor — the blockwise absolute maximum:

$$c_k = \max_{j=0}^{B-1} \left| w_{kB + j} \right|, \qquad k = 0, 1, \ldots, \left\lceil \frac{n}{B} \right\rceil - 1.$$

The resulting vector $c \in \mathbb{R}^{\lceil n/B \rceil}$ contains one scaling
constant per block. Before quantizing (e.g. 4-bit lookup), each block's
elements are divided by its $c_k$ to map them into $[-1, 1]$, where the
quantization grid is defined. Using a per-block scale instead of a single
global scale lets the quantizer allocate more resolution to blocks with small
dynamic range, reducing overall error.

When $n$ is not a multiple of $B$, the final block is shorter and is
zero-padded to length $B$ before the max is taken; the padding does not affect
the result because $|0| = 0$.

## Task

Implement `blockwise_absmax`:

```python
import numpy as np

def blockwise_absmax(w: np.ndarray, block_size: int = 64) -> np.ndarray:
    """Return the per-block maximum absolute value of w.

    Parameters
    ----------
    w : 1-D array of weights (any integer or float dtype).
    block_size : size of each block (default 64).

    Returns
    -------
    c : 1-D float64 array of length ceil(len(w) / block_size).
        c[k] = max |w[k*B : (k+1)*B]|  (last block zero-padded).
    """
```

Use vectorized NumPy — no Python `for` loops. The output must be `float64`.

## Example

```python
import numpy as np
w = np.array([1, -3, 2, 0, -1, 4, 0.5, -0.25])
c = blockwise_absmax(w, block_size=4)
# Block 0: |1|, |-3|, |2|, |0|  → 3.0
# Block 1: |-1|, |4|, |0.5|, |-0.25| → 4.0
# c = array([3., 4.])
```

## What the gate checks

One gate: `max_abs_err`. The grader computes the reference answer using a
NumPy oracle (reshape + `np.max(np.abs(...), axis=1)`) on five random weight
vectors including one whose length is not a multiple of the block size. The
maximum absolute error across all test cases must satisfy
$\text{max\_abs\_err} < 10^{-10}$.
