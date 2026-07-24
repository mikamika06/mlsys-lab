## Context

A serving system has `slot_count` ($S$) concurrent processing slots and a
batch of $N$ requests, all already waiting at $t = 0$, each with a known
generation length $\ell_i$ (its total service time). Whenever a slot is
free, the scheduler dispatches the next request from an **admission
order** into it (non-preemptive, work-conserving, single-server-per-slot):
the request occupies that slot for exactly $\ell_i$ time units, and its
**completion time** is (the time the slot became free for it) $+ \ell_i$.

FIFO admission (dispatch in arrival/submission order) is simple but can be
badly suboptimal: a single very long request dispatched early ties up a
slot and delays every short request behind it in the same slot, inflating
the *mean* completion time across the batch, even though nothing about
FIFO changes any individual request's own length.

For $P$ identical parallel slots serving jobs that are all available at
$t=0$, minimizing the mean (equivalently the sum) of completion times is
solved optimally by dispatching requests in **ascending order of
$\ell_i$** — Shortest-Job-First (SJF) admission — always handing the next
free slot the shortest remaining waiting request. This greedy list
schedule is a classical, provably optimal result for $P \,||\, \sum C_j$.

## Task

Implement `admission_order`:

```python
def admission_order(gen_lens: list[int], slot_count: int) -> tuple[list[int], float]:
    ...
```

- `gen_lens`: a list of `N` positive ints, the generation length of each
  waiting request (index = request id), all present at $t=0$.
- `slot_count`: $S \ge 1$, the number of concurrent processing slots.

Return `(order, mean_completion_latency)`:

- `order`: a permutation of `range(N)` — the admission order in which
  requests are dispatched. Simulate it with the list-scheduling rule
  above: maintain `slot_count` slot-free-times starting at 0; process
  `order` in sequence, each time assigning the request to **whichever
  slot is free soonest**, setting that request's completion time to
  `(that slot's free time) + gen_lens[request]`, and updating the slot's
  free time to that completion time.
- `mean_completion_latency`: the mean of all $N$ completion times
  produced by simulating **your own** `order` exactly as described above
  — not a value computed by some other means.

Your `order` must actually achieve the minimum possible mean completion
latency for the given `gen_lens` and `slot_count`.

## Example

```python
gen_lens = [3, 1, 4, 1, 5]
slot_count = 2

order, mean_lat = admission_order(gen_lens, slot_count)
# SJF dispatch order by length: ids 1,3 (len 1), 0 (len 3), 2 (len 4), 4 (len 5)
# slot A: 1 (0->1), then 0 (1->4), then 4 (4->9)
# slot B: 3 (0->1), then 2 (1->5)
# completions = [4, 1, 5, 1, 9] -> mean = 20/5 = 4.0
# any order that ties up a slot with the length-5 job before the two
# length-1 jobs finish would give a strictly worse (higher) mean.
```

## What the gate checks

The grader builds several `(gen_lens, slot_count)` scenarios from a
seeded NumPy generator (varying request counts, slot counts including
`slot_count=1` and `slot_count >= N`, and tied lengths) and computes the
true SJF-optimal mean completion latency independently: sort `gen_lens`
ascending and run the same slot-free-time simulation described above —
this is the real oracle, provably optimal for this scheduling model,
never calling your function.

For each scenario, the grader also **independently re-simulates your
returned `order`** (never trusting the `mean_completion_latency` you
report) to get your order's true achieved mean, and separately checks
that your reported value matches your own order's true simulated result.
The `rel_err` gate compares your order's true achieved mean against the
oracle's optimum and requires `<= 1e-9` — essentially requiring your
order to actually **be** an SJF-optimal admission order, not just close
to one. Reporting a fabricated `mean_completion_latency`, returning a
non-permutation, or dispatching in any order that isn't (weakly)
non-decreasing in `gen_lens` will fail the gate.
