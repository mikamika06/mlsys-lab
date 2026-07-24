## Context

In many data‑processing pipelines a *batch* is closed either when it reaches a predetermined size or when the oldest element has waited too long. The first mechanism is called a **size trigger**; the second a **timeout trigger**. Formally, let $t_1 < t_2 < \dots < t_m$ be arrival timestamps of items in seconds. A batch starts with an item at time $s$. For each subsequent timestamp $t$, if $t-s \geq T_{\text{max}}$ the current batch is closed by a timeout, otherwise it is added to the batch. If the number of items in the batch reaches $N_{\text{size}}$ before any timeout occurs, the batch is closed by size.

The task is to label each closed batch with 0 for *size* and 1 for *timeout*.

## Task

Implement `classify_batches`:

```python
def classify_batches(timestamps: np.ndarray,
                     batch_size: int,
                     timeout: float) -> np.ndarray:
    ...
```

- `timestamps` is a one‑dimensional NumPy array of strictly increasing floats.  
- The function returns a 1‑D NumPy array of integers (0 or 1), one per closed batch, in the order they were formed.

The implementation must be pure Python/NumPy and run in $O(m)$ time.

## Example

```python
import numpy as np
ts = np.array([0.0, 0.5, 1.2, 3.0, 3.4, 6.0])
labels = classify_batches(ts, batch_size=3, timeout=2.0)
print(labels)   # [0, 1]
```

Explanation:  
- The first three items form a size‑triggered batch (indices 0–2).  
- After the third item, the next timestamp is at 3.0 s; the oldest in the current batch waited $3.0 - 1.2 = 1.8 < 2.0$, so it was added.  
- The following timestamp 3.4 s makes the waiting time $3.4 - 1.2 = 2.2 \ge 2.0$, so the batch closes by timeout before adding 3.4.  
- Item at 6.0 starts a new batch that never reaches size or timeout, but is considered closed by size for simplicity.

## What the gate checks

The grader recomputes the expected labels using an oracle implementation and compares them element‑wise. The solution must produce exactly the same array; otherwise the `exact_match` metric fails.
