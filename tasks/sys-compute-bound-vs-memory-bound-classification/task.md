## Context

The roofline model describes the performance of a kernel on a given machine in terms of its **arithmetic intensity** (AI) and the machine’s **balance point** \(B\).  
Arithmetic intensity is defined as

$$
\text{AI} = \frac{\text{FLOPs}}{\text{Bytes transferred}}
$$

and the balance point is

$$
B = \frac{\text{Peak FLOPs/s}}{\text{Memory bandwidth (Bytes/s)}}
$$

A kernel is **compute‑bound** if its AI exceeds \(B\); otherwise it is **memory‑bound**.  The classification is a simple comparison:

$$
\text{label} =
\begin{cases}
\text{``compute-bound''} & \text{if } \text{AI} > B,\\[4pt]
\text{``memory-bound''} & \text{otherwise}.
\end{cases}
$$

## Task

Implement the function `classify_bound` that takes a 1‑D NumPy array of arithmetic intensities and a scalar balance point, and returns an array of strings with the same shape containing either `"compute-bound"` or `"memory-bound"` for each kernel.

```python
import numpy as np

def classify_bound(ai: np.ndarray, balance: float) -> np.ndarray:
    ...
```

The function must use NumPy only; no Python loops are required.  The returned array should be of type `np.str_` (or equivalent).

## Example

```python
import numpy as np
ai = np.array([0.5, 1.2, 3.4])
balance = 1.0
labels = classify_bound(ai, balance)
print(labels)   # ['memory-bound', 'compute-bound', 'compute-bound']
```

## What the gate checks

The grader computes a reference classification using the same rule and compares it to your output with an exact match metric.  The candidate must return the correct labels for all test cases; any mismatch yields a failure.
