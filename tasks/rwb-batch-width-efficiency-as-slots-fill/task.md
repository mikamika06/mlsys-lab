## Context

A continuous-batching inference server serves up to $N$ requests concurrently
— $N$ is the configured **batch width**. At every scheduling step $t$ some
number of slots $b_t \in \{0, 1, \dots, N\}$ are occupied by in-flight
requests (the rest are idle, waiting for new work to fill them). As requests
join and finish, $b_t$ fluctuates step to step.

The **per-step utilization** is the fraction of slots actually busy,

$$
u_t = \frac{b_t}{N}, \qquad t = 0, \dots, T-1,
$$

and the **mean utilization** over the trace is

$$
\bar u = \frac{1}{T}\sum_{t=0}^{T-1} u_t .
$$

$\bar u$ close to $1$ means the batch stays nearly full (good throughput);
$\bar u$ far below $1$ means slots regularly sit idle while ramping up or
draining down.

## Task

Implement `batch_width_utilization`:

```python
def batch_width_utilization(occupancy: np.ndarray, N: int) -> dict:
    ...
```

- `occupancy` — 1-D integer NumPy array of length $T$; `occupancy[t]` is the
  number of busy slots $b_t$ at step $t$, with $0 \le b_t \le N$.
- `N` — the configured batch width (a positive integer).

Return a `dict` with:

- `"per_step"` — a `float64` NumPy array of shape $(T,)$: $u_t = b_t / N$.
- `"mean"` — a Python `float`: $\bar u$, the mean of `"per_step"`.

## Example

```python
import numpy as np

occupancy = np.array([2, 4, 4, 3])
N = 4

result = batch_width_utilization(occupancy, N)
print(result["per_step"])  # [0.5, 1.0, 1.0, 0.75]
print(result["mean"])      # 0.8125
```

## What the gate checks

The grader builds several random occupancy traces (with varying $T$ and $N$,
occupancy always between $0$ and $N$) and computes the reference
$u_t = b_t/N$ and $\bar u$ directly with NumPy. Your `"per_step"` array and
`"mean"` value are concatenated into one vector and compared against the
reference with relative L2 error:

$$
\mathrm{rel\_err} = \frac{\lVert \hat v - v \rVert_2}{\lVert v \rVert_2} < 10^{-9}.
$$

Returning raw slot counts instead of dividing by $N$, or computing the mean
over the wrong axis or length, will fail the gate.
