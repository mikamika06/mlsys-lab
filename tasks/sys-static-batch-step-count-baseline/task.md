## Context

In many language‑model inference pipelines, a *static batch* groups incoming requests into fixed‑size batches and decodes them together. During decoding each request consumes one token per step; the batch proceeds for as many steps as the longest request in that batch. If the request lengths are $L_1,\dots,L_n$ and the batch size is $B$, then the total number of decoding steps is

$$
S = \sum_{k=0}^{\lceil n/B\rceil-1} \max_{i=kB}^{(k+1)B-1} L_i .
$$

This metric captures the computational cost of a static‑batch scheduler.

## Task

Implement `static_batch_steps(request_lengths, batch_size)`:

```python
def static_batch_steps(request_lengths: np.ndarray, batch_size: int) -> np.ndarray:
    ...
```

* `request_lengths` is a 1‑D NumPy array of positive integers (token counts).
* `batch_size` is a positive integer.
* The function must return a **scalar** NumPy array of dtype `int64` containing the total number of decoding steps $S$.

The implementation should use only NumPy operations; no explicit Python loops over requests are allowed.

## Example

```python
import numpy as np
reqs = np.array([5, 3, 7])
steps = static_batch_steps(reqs, batch_size=2)
print(steps)          # array(12, dtype=int64)
```

The batches are `[5, 3]` and `[7]`; the steps are $\max(5,3)+\max(7)=5+7=12$.

## What the gate checks

Two gates evaluate a candidate solution:

1. **exact_match** – The returned value must equal the analytically computed total steps.
2. **size_ratio** – The output array’s byte size must match that of the reference array; this guarantees the correct shape and dtype.

Both metrics must satisfy their thresholds for the submission to pass.
