## Context

Continuous batching schedulers may preempt a running sequence when a higher-priority request arrives. Two common choices are swapping the paused state to host memory or discarding it and recomputing the state when the sequence resumes.

Assume a request has a key-value cache containing $t$ tokens and each token occupies $b$ bytes. The cache size is measured by the actual byte size of a NumPy `uint8` buffer:

$$
\mathrm{cache\_bytes}(t,b)=\mathrm{nbytes}(\mathrm{uint8}[t \times b]).
$$

For a pause/resume pair, swapping moves the cache out and back:

$$
C_{\mathrm{swap}} = 2 \times \mathrm{cache\_bytes}(t,b).
$$

Recompute discards the cache on pause and rebuilds it during resume:

$$
C_{\mathrm{recompute}} = \mathrm{cache\_bytes}(t,b).
$$

A trace contains iteration-level scheduling events. Each pause event creates a pending preemption cost that is charged when the matching resume event occurs.

## Task

Implement `modeled_mem_access(trace, bytes_per_token)`:

```python
def modeled_mem_access(trace: list[tuple], bytes_per_token: int) -> dict:
    ...
```

The input `trace` contains tuples:

- `("pause", request_id, token_count)`
- `("resume", request_id)`

For every paused request that later resumes, compute the total modeled bytes moved by the recompute strategy and the swap strategy.

Return:

```python
{
    "recompute_bytes": int,
    "swap_bytes": int,
}
```

Use the token count from the matching pause event. Ignore resumes without a matching pause.

## Example

```python
trace = [
    ("pause", "a", 10),
    ("resume", "a"),
]

modeled_mem_access(trace, 4)
# {
#   "recompute_bytes": 40,
#   "swap_bytes": 80,
# }
```

## What the gate checks

The gate runs several scheduling traces and computes the expected byte counts using a NumPy `uint8` allocation as the byte-size oracle. The returned dictionary must exactly match the oracle result for every trace.
