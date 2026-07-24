## Context

In many language‑model inference pipelines a *budget* of tokens is allocated per decoding step.  
A fixed number of **decode** tokens are reserved first; any remaining budget is used for a **prefill** chunk that can be processed in parallel.  

Let $b_i$ denote the token budget at step $i$, and let $d_i$ be the number of decode tokens required at that step.  
The prefill allocation is then

$$
p_i = \max\!\bigl(0,\; b_i - d_i\bigr).
$$

If $d_i > b_i$ no prefill tokens are available for that step.

The task is to implement a vectorised routine that, given arrays of budgets and decode counts, returns the per‑step prefill allocation and a boolean mask indicating whether any prefill tokens were allocated at each step.

## Task

Implement `classify_prefill_decode`:

```python
def classify_prefill_decode(budgets: np.ndarray,
                            decode_counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ...
```

* `budgets`: 1‑D NumPy array of non‑negative integers.
* `decode_counts`: 1‑D NumPy array of the same shape as `budgets`.
* Return a tuple `(prefill_tokens, is_prefill)` where
  * `prefill_tokens` is an integer array of the same shape containing $p_i$.
  * `is_prefill` is a boolean array that is `True` iff $p_i > 0$.

The implementation must be fully vectorised (no explicit Python loops) and should raise a `ValueError` if the input shapes differ.

## Example

```python
import numpy as np
budgets = np.array([10, 5, 8])
decode_counts = np.array([3, 7, 2])

prefill, is_prefill = classify_prefill_decode(budgets, decode_counts)
print(prefill)      # [7 0 6]
print(is_prefill)   # [ True False  True]
```

## What the gate checks

The grader computes a reference solution using NumPy vectorised operations and compares your output **exactly**.  
If any element differs or if shapes mismatch, the `exact_match` metric is set to `0.0`; otherwise it is `1.0`. No other metrics are evaluated.
