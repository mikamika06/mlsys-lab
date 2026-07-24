## Context

In distributed training and other parallel workloads, a worker typically performs two kinds of work in each step: **compute** (e.g. forward/backward passes) and **communication** (e.g. gradient exchange).  
If the compute and communication can be overlapped perfectly, the total time spent on a step is simply

$$T_{\text{overlap}} = \max\!\bigl(\sum_i c_i,\;\sum_i m_i\bigr),$$

where $c_i$ are per‑step compute times and $m_i$ are per‑step communication times.  
The **serial** time, where the two phases run one after another, is

$$T_{\text{serial}} = \sum_i c_i + \sum_i m_i.$$

A small relative error between a candidate implementation and this ideal model indicates that the implementation correctly captures the overlap behaviour.

## Task

Implement `overlap_time(compute_times, comm_times)`:

```python
def overlap_time(compute_times: np.ndarray,
                 comm_times: np.ndarray) -> float:
    ...
```

* `compute_times` and `comm_times` are 1‑D NumPy arrays of the same length containing positive floating‑point values.  
* The function must return a single Python `float` (dtype `float64`) equal to $T_{\text{overlap}}$ as defined above.  
* Use only vectorised NumPy operations; no explicit Python loops.

## Example

```python
import numpy as np
compute = np.array([2.0, 3.5])
comm    = np.array([4.0, 1.0])

T_overlap = overlap_time(compute, comm)
print(T_overlap)          # 5.5
```

Here $\sum c_i = 5.5$, $\sum m_i = 5.0$, so $T_{\text{overlap}}=\max(5.5,5.0)=5.5$.

## What the gate checks

Two metrics are evaluated:

1. **Relative error** `rel_err` – the global relative L2 error between your result and a NumPy reference must satisfy  
   $$\mathrm{rel\_err} \le 10^{-9}.$$

2. **Operation count** `op_count` – the number of Python line events executed inside your function, recorded by a tracer, must be at most 50.  

A fully vectorised solution will pass both gates.
