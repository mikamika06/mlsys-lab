## Context

ZeRO-3 training shards model parameters across workers. During execution, parameters for a layer can be prefetched before they are needed. Communication for a layer is hidden when it completes during the available compute window created by earlier layers.

This task uses a simplified scheduling model. For layer $i$, let $B_i$ be the bytes communicated to materialize its parameters and let $T_i$ be its compute time. The communication duration is

$$
C_i = \frac{B_i}{R},
$$

where $R$ is the communication bandwidth in bytes per unit time.

A prefetch depth $k$ means layer $i$ can have its communication start $k$ compute steps before layer $i$ executes. The communication is fully hidden if the previous $k$ compute times provide enough overlap:

$$
\sum_{j=\max(0,i-k)}^{i-1} T_j \ge C_i .
$$

Otherwise, the remaining exposed communication is the uncovered portion:

$$
E_i = \max\left(0, C_i - \sum_{j=\max(0,i-k)}^{i-1} T_j\right) R .
$$

The total exposed communication bytes are

$$
E = \sum_i E_i .
$$

## Task

Implement `exposed_comm_bytes(layer_bytes, compute_times, bandwidth, prefetch_depth)`.

The arguments are:

- `layer_bytes`: a sequence of parameter communication sizes $B_i$ in bytes.
- `compute_times`: a sequence of compute durations $T_i$.
- `bandwidth`: communication bandwidth $R$ in bytes per unit time.
- `prefetch_depth`: the number of previous compute intervals available for overlap.

Return the exposed communication bytes as a Python `float`.

The input lengths are equal. `prefetch_depth` is a non-negative integer. Use the scheduling model from the context exactly.

## Example

```python
bytes_per_layer = [1000, 2000, 3000]
compute = [1.0, 0.5, 2.0]

answer = exposed_comm_bytes(
    bytes_per_layer,
    compute,
    bandwidth=1000,
    prefetch_depth=1,
)

# Layer communication:
# 1000 bytes: exposed fully
# 2000 bytes: 1000 bytes hidden by layer 0 compute, 1000 exposed
# 3000 bytes: 500 bytes hidden by layer 1 compute, 2500 exposed
# answer == 4500.0
```

## What the gate checks

The gate builds several scheduling cases and compares the submitted implementation against an oracle implementation of the scheduling equations. The metric `modeled_mem_access` is `1.0` only when every case returns the same exposed communication bytes as the model.
