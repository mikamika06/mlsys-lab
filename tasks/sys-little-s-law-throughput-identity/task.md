## Context

Little's Law is a fundamental relationship in queuing theory that links the average number of items in a system (concurrency), the throughput, and the mean latency. It states that

$$\text{throughput} = \frac{\text{concurrency}}{\text{latency}}\;.$$

In many performance‑analysis scenarios we are given measurements of concurrency and mean latency for a set of workloads and must compute the corresponding throughput values.

## Task

Implement `compute_throughput(concurrency, latency)`:

```python
def compute_throughput(concurrency: np.ndarray,
                       latency: np.ndarray) -> np.ndarray:
    ...
```

The function receives two 1‑D NumPy arrays of equal length.  
`concurrency[i]` is the average number of concurrent requests for workload *i*, and `latency[i]` is the mean service time (in seconds).  
Return a NumPy array containing the throughput for each workload, computed as

$$\text{throughput}_i = \frac{\text{concurrency}_i}{\text{latency}_i}\;.$$

The result must be of type `float64` and have the same shape as the inputs.

## Example

```python
import numpy as np
from compute_throughput import compute_throughput

concurrency = np.array([10, 20, 30], dtype=np.float64)
latency     = np.array([0.5, 1.0, 2.0], dtype=np.float64)

throughput = compute_throughput(concurrency, latency)
print(throughput)   # [20. 20. 15.]
```

## What the gate checks

The grader computes a reference throughput using NumPy division and compares it to your output with the global relative L2 error

$$\text{rel\_err} = \frac{\lVert \hat y - y\rVert}{\lVert y\rVert}\;.$$

Your solution must achieve $\text{rel\_err}\le 10^{-9}$ on all automatically generated test cases. The gate will fail if the error exceeds this threshold or if your function does not return a NumPy array of type `float64`.
