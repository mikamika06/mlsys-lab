## Context

A continuous batching scheduler keeps a fixed number of execution slots. Each request
has a generation length $L$ and occupies a slot for $L$ iterations. When a request
finishes, its slot must be retired immediately so a waiting request can be admitted.

The scheduler state at iteration $t$ can be represented by the active sequence IDs:

$$S_t = [q_1, q_2, \dots, q_k]$$

where $k$ is the number of occupied slots and the order is the slot order. A correct
scheduler updates the active set after each iteration:

1. Run one generation step for all active requests.
2. Increment their generated token counts.
3. Remove requests where generated tokens reach their generation length.
4. Admit waiting requests into newly free slots.

The retirement step is essential because the maximum number of active requests is
limited by the number of slots $C$.

## Task

Implement `schedule_trace(reqs, slots)`.

`reqs` is a list of pairs `(request_id, gen_len)` in arrival order. `request_id`
is a unique integer and `gen_len` is a positive integer. `slots` is the maximum
number of concurrent requests.

The function must return a list where each element is the active sequence IDs after
each scheduler iteration. The returned IDs must appear in slot order.

A request is admitted when a slot is available and remains active until its generated
token count becomes equal to `gen_len`. Finished requests must not remain in the
trace after the iteration where they complete.

The scheduler starts with no active requests and admits requests before the first
iteration. Continue until both the waiting queue and active slots are empty.

## Example

```python
trace = schedule_trace([(10, 2), (20, 1), (30, 2)], 2)

# [
#   [10, 20],
#   [10, 30],
#   [30],
#   []
# ]
```

Request `20` finishes during the first iteration, so request `30` can enter the
second slot on the next iteration.

## What the gate checks

The gate compares the returned active-set trace against a reference continuous
scheduler simulation. The exact sequence of active IDs must match the oracle output
for several cases. A scheduler that leaves completed requests in their slots will
fail because waiting requests will not be admitted at the correct time.
