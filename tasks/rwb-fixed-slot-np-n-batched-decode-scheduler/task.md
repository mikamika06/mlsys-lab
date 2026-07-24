## Context

`llama.cpp`'s server runs with a fixed number of parallel decode slots
(`--parallel N` / `-np N`): $N$ physical slots, each holding exactly one
request from the moment it is **admitted** until it finishes generating
— unlike vLLM-style continuous batching, a slot is never shared or
preempted mid-generation. When a request finishes, its slot becomes free
and the next waiting request is admitted into it, in FIFO arrival order.

At each discrete decode step $t$:

1. Free every slot whose occupant already generated its full `gen_len`
   tokens as of the end of the previous step.
2. Move every request that has arrived (`arrival_step \le t`) and isn't
   admitted yet into the waiting queue, in FIFO order (`arrival_step`,
   ties broken by request id).
3. Fill any free slots from the front of the waiting queue.
4. Every occupied slot decodes exactly one token this step.
5. Record which request occupies each slot this step (or "idle").

A slot that reaches `gen_len` tokens *this* step still shows that
request as occupying it in this step's snapshot — it's freed starting
the *next* step.

## Task

Implement `slot_occupancy_trajectory`:

```python
def slot_occupancy_trajectory(reqs: list[tuple[int, int]], N: int) -> list[list[int]]:
    ...
```

- `reqs`: list of `(arrival_step, gen_len)` pairs; request `i`'s id is
  its index into `reqs`.
- `N`: number of fixed decode slots.
- Simulate exactly the 5-step recurrence above, starting at `t = 0`.
- Return a list of steps (one entry per decode step until every request
  has finished — no trailing all-idle step after that), each a length-`N`
  list giving the request id occupying that slot (`-1` if idle).

## Example

```python
reqs = [(0, 3), (0, 2)]   # two requests, both arrive at t=0
slot_occupancy_trajectory(reqs, N=1)
# [[0], [0], [0], [1], [1]]
# request 0 (gen_len 3) occupies the only slot for steps 0-2; request 1
# only gets admitted once it frees, at step 3, and runs steps 3-4.
```

## What the gate checks

The grader loads a committed fixture — 10 requests with staggered
arrivals and a mix of short and long generations under `N = 3` — plus
several additional seeded random request streams and slot counts, and
replays the same 5-step simulation independently in Python (never
calling your function, never hardcoding an expected trajectory).

`exact_match` is the fraction of step-snapshots, across all cases, that
match the oracle's exactly (every one of the `N` slot values). The gate
requires `1.0`. Admitting out of arrival order, freeing a slot one step
too early or late, or forgetting to keep decoding every already-busy
slot every step will all show up as a mismatched slot value somewhere in
the trajectory.
