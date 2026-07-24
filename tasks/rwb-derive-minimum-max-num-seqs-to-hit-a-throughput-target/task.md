## Context

Production inference schedulers expose a concurrency limit such as `max_num_seqs`.
A higher value allows more requests to overlap, but increasing it past the useful
range does not improve the completion time of a fixed trace.

For a trace of requests, a continuous simulation can model the system as a set of
identical execution slots. Each request has a service time $t_i$. With a
concurrency limit $S$, requests are assigned to the first available slot in
arrival order. The makespan $M(S)$ is the completion time of the last request.

The goal is to find the smallest concurrency value that satisfies a throughput
target:

$$
S^* = \min \{S \in \mathbb{Z}_{>0} : M(S) \leq M^*\}.
$$

If no concurrency value up to the number of requests can satisfy the target, the
answer is $-1$.

## Task

Implement `minimum_max_num_seqs(reqs, target_makespan)`:

```python
def minimum_max_num_seqs(reqs, target_makespan):
    ...
```

`reqs` is a list of dictionaries. Each dictionary contains:

- `arrival`: request arrival time as a float.
- `prefill`: prefill service time as a float.
- `decode`: decode service time as a float.

The service time is:

$$
t_i = \mathrm{prefill}_i + \mathrm{decode}_i .
$$

Return the smallest positive integer `S` such that the continuous simulation
makespan is less than or equal to `target_makespan`. Return `-1` when the target
cannot be reached even with one slot per request.

The simulation uses the following policy for a fixed $S$:

1. Process requests in increasing arrival order.
2. Assign each request to the slot with the earliest available time.
3. A request starts at the later of its arrival time and that slot's available time.
4. The slot becomes available at start time plus the request service time.

Do not modify the input list.

## Example

```python
reqs = [
    {"arrival": 0.0, "prefill": 2.0, "decode": 3.0},
    {"arrival": 1.0, "prefill": 1.0, "decode": 1.0},
    {"arrival": 2.0, "prefill": 4.0, "decode": 1.0},
]

answer = minimum_max_num_seqs(reqs, 8.0)
# answer == 2
```

## What the gate checks

The grader builds a reference answer by running the same continuous scheduling
oracle and sweeping concurrency values upward. The returned integer must exactly
match the oracle result.
