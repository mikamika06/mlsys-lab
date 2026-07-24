## Context

In many data‑processing pipelines a *captured* batch is one whose size does not exceed a pre‑defined bucket limit. Batches larger than this limit are processed eagerly and bypass the captured path. The decision rule is simply an inequality on the integer batch size.

Let $B_{\max}$ denote the maximum size that can be captured, and let $s_i$ be the size of the *i*‑th incoming batch. Then

$$
\text{captured}(s_i) = 
\begin{cases}
\texttt{True} & \text{if } s_i \le B_{\max},\\[4pt]
\texttt{False} & \text{otherwise}.
\end{cases}
$$

The task is to implement this rule efficiently for a whole list of batch sizes.

## Task

Implement `classify_batches(max_bucket, batch_sizes)`:

```python
def classify_batches(max_bucket: int, batch_sizes: Iterable[int]) -> np.ndarray:
    ...
```

It receives an integer $B_{\max}$ and an iterable (list or NumPy array) of non‑negative integers. It must return a one‑dimensional NumPy array of dtype `bool` where each element is the result of the rule above for the corresponding batch size.

The implementation should be fully vectorised; no explicit Python loops are required.

## Example

```python
import numpy as np
max_bucket = 10
sizes = [5, 12, 10]
labels = classify_batches(max_bucket, sizes)
print(labels)          # [ True False  True]
```

## What the gate checks

The grader recomputes the classification using a reference implementation and compares the returned array with `np.array_equal`. The metric `exact_match` must be exactly `1.0`; any deviation yields `0.0`.

No additional performance or size constraints are imposed beyond correctness.
