## Context

In production inference servers, requests arrive continuously and are batched to exploit parallelism.  
A standard batching policy dispatches a batch when either:

* the number of accumulated requests reaches a maximum `max_batch_size`, or  
* the time since the first request in the batch reaches a timeout `batch_timeout`.

The sequence of dispatched batch sizes forms a distribution that system operators monitor.  
The *achieved‑batch‑size histogram* is a diagnostic that reveals whether the system is operating near its batch‑size limit or frequently dispatching small batches.

Let $t_1, t_2, \ldots, t_N$ be a sorted sequence of request arrival timestamps.  
The simulation proceeds by scanning arrivals in order:

1. If the batch is empty, record the current timestamp as $t_{\text{start}}$.  
2. Append the request to the pending queue (queue length increments).  
3. If the queue length equals `max_batch_size` or  
   $t_i - t_{\text{start}} \geq \text{batch\_timeout}$, dispatch the batch:  
   increment the histogram bin corresponding to the queue length, clear the queue,  
   and reset $t_{\text{start}}$ to $\text{None}$.  
4. After all arrivals, if the queue is non‑empty, dispatch the final batch.

The histogram $H$ is a vector of length $\text{max\_batch\_size}+1$ (dtype int) where  
$H[s]$ counts dispatched batches of exactly $s$ requests.

## Task

Implement:

```python
def batch_size_histogram(arrivals, batch_timeout, max_batch_size):
    """Return the histogram of formed batch sizes (length max_batch_size+1)."""
```

* `arrivals`: 1‑D list[float] of shape $(N,)$ with strictly increasing float timestamps.  
* `batch_timeout`: positive float, time window in seconds.  
* `max_batch_size`: positive integer, maximum requests per batch.  
* **Returns:** 1‑D list[float] of shape $(\text{max\_batch\_size}+1,)$, integer counts.

The algorithm must scan arrivals exactly once and manage state explicitly.

## Example

```python
arrivals = [0.0, 0.5, 1.2, 1.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]
hist = batch_size_histogram(arrivals, batch_timeout=1.0, max_batch_size=4)
print(hist)  # [0, 0, 2, 2, 0]
```

Batches formed:

* At $t=1.2$: first three requests ($0.0, 0.5, 1.2$) timeout → size $3$  
* At $t=3.0$: requests $1.5, 3.0$ timeout → size $2$  
* At $t=4.5$: requests $3.5, 4.0, 4.5$ timeout → size $3$  
* At $t=6.0$: requests $5.0, 6.0$ timeout → size $2$

Histogram: two batches of size $2$, two of size $3$.

## What the gate checks

The function is tested on exactly one arrivals trace with  
`batch_timeout = 1.5` and `max_batch_size = 8`.  
Your histogram must match the reference simulation exactly (all entries equal).  
There is no tolerance: the histogram counts are integers, so the gate requires a perfect match.

A correct implementation scans the array in one pass and manages the batch state correctly.
