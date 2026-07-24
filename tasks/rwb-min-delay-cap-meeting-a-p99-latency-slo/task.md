## Context

The dynamic batcher from before dispatches whenever EITHER trigger fires
first: the queue reaches `preferred_size` (size trigger), or the oldest
pending item has waited `max_queue_delay` = $C$ time units (delay
trigger, dispatching the whole current — possibly partial — queue). At a
tie, the size trigger wins.

Every request's end-to-end latency is its queue wait plus a fixed
per-batch compute cost:

$$
\text{latency}_j = \big(t^{\text{formation}}_j - t^{\text{arrival}}_j\big) + \text{service\_time}
$$

where $t^{\text{formation}}_j$ is the time request $j$'s batch was
dispatched. The tail latency SLO is measured with the **nearest-rank**
p99 (the smallest latency such that at least 99% of requests are at or
below it):

$$
p_{99} = \text{sorted\_latencies}\big[\lceil 0.99\,n \rceil - 1\big]
\qquad (\text{0-indexed}, \ n = \text{number of requests})
$$

Operators want the **smallest** delay cap $C$ — from a fixed list of
candidates — whose resulting p99 still meets the SLO: a smaller cap
flushes partial batches sooner (better latency, but smaller, less
efficient batches); the smallest cap that still clears the SLO is the
best trade-off.

## Task

Implement `min_cap_meeting_slo`:

```python
def min_cap_meeting_slo(arrivals: list[int], candidate_caps: list[int],
                         preferred_size: int, service_time: int, slo: float) -> int:
    ...
```

- `arrivals`: list of int arrival times (not necessarily sorted).
- `candidate_caps`: list of positive int candidate `max_queue_delay`
  values (not necessarily sorted; may contain duplicates).
- `preferred_size`: positive int, the batcher's size trigger (fixed
  across all candidates).
- `service_time`: non-negative int, fixed compute time added to every
  request's latency.
- `slo`: float, the p99 latency budget.

For each candidate cap, in ascending order **by value**, simulate the
two-trigger batcher over `arrivals`, compute every request's latency,
and its p99 as defined above. Return the first (smallest) cap whose p99
is `<= slo`. If none of the candidates meet the SLO, return `-1`.

## Example

```python
arrivals = [0, 1, 1, 2, 10, 11, 12, 13, 30, 45, 46, 70, 71, 72, 73, 74, 100, 140, 141, 200]
candidate_caps = [50, 30, 2, 20, 10, 15, 5]   # deliberately unsorted
preferred_size = 4
service_time = 5
slo = 9.0

min_cap_meeting_slo(arrivals, candidate_caps, preferred_size, service_time, slo)
# 2  -- cap=2 already gives p99=7 <= 9 (the smallest candidate, tried
#        first once sorted by value); larger caps only ever produce a
#        p99 >= the smaller ones' here, since a bigger cap can only
#        delay a partial flush further, never speed one up.
```

## What the gate checks

The grader loads a committed `arrivals.npy` fixture, graded with the
exact `candidate_caps`, `preferred_size`, `service_time`, `slo` from the
example above, plus several additional seeded random scenarios (varying
arrival streams, candidate lists, and SLOs, including cases where no
candidate meets the SLO), and computes the answer with an independent
implementation of the same batcher simulation, latency formula, and
nearest-rank p99 — never calling your function, never hardcoding an
expected cap.

`exact_match` is the fraction of cases where your returned integer
exactly equals the oracle's, and must be `1.0`. Sweeping candidates in
their given list order instead of sorted by value, using a different p99
convention (e.g. linear-interpolation percentile instead of nearest-rank),
forgetting `service_time`, or reusing the wrong tie-break in the
underlying batcher simulation will all pick the wrong cap on at least
one case.
