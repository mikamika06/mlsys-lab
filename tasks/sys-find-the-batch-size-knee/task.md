## Context

In many machine‑learning workloads the throughput (samples per second) and latency (time to process a batch) vary with the chosen batch size.  
For a given system we often have a *latency SLO* (service‑level objective), e.g. a maximum acceptable latency of $L_{\text{SLO}}$.  
The **knee** is defined as the largest batch size that still satisfies the SLO and yields the highest throughput among all such batches.

Mathematically, given arrays

$$
\mathbf{lat} = [l_0,\dots,l_{n-1}], \qquad
\mathbf{thr} = [t_0,\dots,t_{n-1}]
$$

sorted by increasing batch size, the knee index $k$ is

$$
k = \arg\max_{\substack{i \\ l_i \le L_{\text{SLO}}}} t_i,
$$

with the convention that if no $i$ satisfies $l_i \le L_{\text{SLO}}$, then $k=-1$.

## Task

Implement `find_batch_size_knee`:

```python
def find_batch_size_knee(latencies: np.ndarray, throughputs: np.ndarray, slo_latency: float) -> int:
    ...
```

The function receives two one‑dimensional NumPy arrays of equal length and a scalar SLO latency.  
It must return the integer index of the knee batch size as defined above.

## Example

```python
import numpy as np
lat = np.array([10, 12, 15, 20])
thr = np.array([100, 120, 110, 90])
slo = 18.0
knee = find_batch_size_knee(lat, thr, slo)
print(knee)   # → 1
```

The first three batches satisfy the latency SLO; among them batch 1 has the highest throughput.

## What the gate checks

A single gate named `exact_match` verifies that the returned index equals the reference value computed by an oracle. No other metrics are required.
