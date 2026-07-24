## Context

Continuous batching schedules inference requests at iteration boundaries instead of waiting
for an entire batch to finish. At each decoding step, active requests produce one token,
completed requests leave the batch, and waiting requests can join the next step.

A request $r$ has a remaining token trace

$$T_r = (t_1, t_2, \dots, t_k),$$

where $k$ is the number of decoding iterations required before the request finishes.
A scheduler maintains an active set $A_i$ at step $i$. The emitted token for a request is
the next element of its trace when that request is active.

The important property is that finished requests release capacity immediately. If the
maximum batch size is $B$, the scheduler should keep adding waiting requests whenever
there is free capacity at the beginning of a decoding step.

## Task

Implement `schedule_decode(requests, batch_size)`:

```python
def schedule_decode(requests, batch_size):
    ...
```

`requests` is a list of pairs `(request_id, tokens)`. `request_id` is an integer and
`tokens` is a list of integers representing the deterministic decode trace for that
request.

Return a dictionary mapping each `request_id` to the list of tokens emitted for that
request.

The scheduler must simulate iteration-level continuous batching:

1. Waiting requests are admitted in input order until the active set reaches
   `batch_size`.
2. Each decoding iteration emits one token for every active request.
3. Requests that emitted their final token leave the active set after that iteration.
4. Newly freed slots are filled by waiting requests before the next iteration.

All requests finish exactly when their token trace has been emitted.

## Example

```python
requests = [
    (10, [4, 5, 6]),
    (11, [7]),
    (12, [8, 9]),
]

out = schedule_decode(requests, 2)

# out == {
#     10: [4, 5, 6],
#     11: [7],
#     12: [8, 9],
# }
```

With a batch size of $2$, requests `10` and `11` start together. After request `11`
finishes, request `12` joins before the next decoding iteration.

## What the gate checks

The gate compares the returned token streams against a reference scheduler that performs
the continuous batching simulation directly. The output must exactly match the oracle
for several traces containing staggered completion times, joins, and capacity limits.

A scheduler that waits for the whole batch to finish, drops late requests, or changes
admission order will fail the exact-match gate.
