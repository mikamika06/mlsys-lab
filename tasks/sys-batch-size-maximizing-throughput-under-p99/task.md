## Context

Batching combines multiple requests into one execution. Larger batches often improve throughput because fixed overhead is amortized, but latency usually increases because a batch must wait for and process more items.

A simple modeled p99 latency curve is:

$$
L_{p99}(b) = L_0 + L_1 b + J b^2 ,
$$

where $b$ is the batch size, $L_0$ is fixed overhead, $L_1$ is the linear processing cost, and $J$ models tail latency growth.

Throughput is the number of items completed per second:

$$
T(b) = \frac{1000 b}{L_{p99}(b)}
$$

when latency is measured in milliseconds. The goal is to choose the integer batch size with the highest throughput while satisfying a latency SLO:

$$
L_{p99}(b) \leq S .
$$

The best batch size is not always the largest allowed batch size because the quadratic tail term can make throughput decrease after a certain point.

## Task

Implement `max_throughput_batch_size`:

```python
def max_throughput_batch_size(
    max_batch: int,
    slo_ms: float,
    fixed_ms: float,
    per_item_ms: float,
    jitter_ms: float,
) -> int:
    ...
```

Return the integer batch size $b$ in the range $1 \leq b \leq$ `max_batch` that maximizes the modeled throughput while keeping p99 latency below or equal to `slo_ms`.

Use the latency model:

$$
L_{p99}(b) = \text{fixed\_ms} + \text{per\_item\_ms} \cdot b + \text{jitter\_ms} \cdot b^2 .
$$

If no batch size satisfies the SLO, return `1`.

If multiple batch sizes have exactly the same throughput, return the smaller batch size.

## Example

```python
b = max_throughput_batch_size(
    max_batch=64,
    slo_ms=50.0,
    fixed_ms=2.0,
    per_item_ms=0.5,
    jitter_ms=0.005,
)
# b is the integer batch size with the highest valid throughput
```

## What the gate checks

The gate recomputes the optimum by exhaustively evaluating the mathematical latency model inside the grader. It compares the throughput achieved by the submitted implementation against that oracle result.

The returned batch size must have an optimum throughput ratio within $1.02$:

$$
\frac{T(b_{\mathrm{oracle}})}{T(b_{\mathrm{submitted}})} \leq 1.02 .
$$

Implementations that only select the largest valid batch size can fail because the throughput curve may peak before the latency boundary.
