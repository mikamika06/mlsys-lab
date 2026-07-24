## Context

A continuous-batching server is launched with a hard concurrency limit
$N$ (the `-np` / `--max-num-seqs` slot count): at every scheduler
iteration, the number of simultaneously active requests must never exceed

$$
|S_t| \le N \qquad \text{for every iteration } t,
$$

where $S_t$ is the set of active request IDs at iteration $t$. Requests
arrive in a queue and are admitted into free slots; each admitted request
occupies its slot for `gen_len` iterations and then retires, freeing the
slot for the next waiting request.

The provided scheduler has an off-by-one bug in its admission check: it
compares the active-slot count with `<=` instead of `<` against $N$,
so it will happily admit *one more* request even when all $N$ slots are
already occupied, producing $N+1$ simultaneously active requests for that
iteration — silently exceeding the hard concurrency limit the caller asked
for.

## Task

Fix `bounded_slot_schedule(reqs, n_slots)`:

```python
def bounded_slot_schedule(reqs, n_slots):
    ...
```

- `reqs`: a list of `(request_id, gen_len)` pairs in arrival order.
  `request_id` is a unique integer; `gen_len` is a positive integer number
  of iterations the request needs.
- `n_slots`: the hard concurrency limit $N$.

Simulate the scheduler iteration by iteration:

1. Admit waiting requests into free slots, in arrival order, **only**
   while fewer than `n_slots` requests are currently active.
2. Run one generation step for every active request (increment its
   consumed-iteration count).
3. Retire any request whose consumed-iteration count reaches its
   `gen_len` immediately, before the next iteration's admission step.

Return a list where entry $t$ is the list of active request IDs (in slot
order) during iteration $t$. The active-request count at every iteration
must satisfy `len(S_t) <= n_slots` — never `n_slots + 1`.

## Example

```python
trace = bounded_slot_schedule([(10, 2), (20, 1), (30, 2)], 2)

# [
#   [10, 20],
#   [10, 30],
#   [30],
# ]
```

Request `20` finishes during the first iteration, freeing a slot so
request `30` can be admitted on the second iteration — at no point are
there 3 active requests, even though there are 3 total requests and the
buggy `<=` check would have let all of them in on the first iteration.

## What the gate checks

The gate re-simulates the same scheduler correctly (admission gated by
`< n_slots`, immediate retirement) and compares the full per-iteration
active-ID trace against your output, exactly, for several request lists
and slot counts. The buggy `<=` admission check causes at least one
iteration to contain `n_slots + 1` active IDs, which will never match the
oracle's trace.
