## Context

A dynamic batcher (as in Triton's request batching) queues incoming
requests and dispatches the whole current queue together as soon as
either trigger fires, whichever happens first:

- **Size trigger.** The queue reaches `preferred_batch_size`: dispatch a
  full batch immediately.
- **Delay trigger.** The oldest item still queued has been waiting
  `max_queue_delay` time units: dispatch the entire current queue right
  away, even if it's smaller than `preferred_batch_size` (a *partial*
  batch — without this, a queue that never quite fills up would never
  flush).

If arrival $a_j$ is the oldest item currently queued, its delay-trigger
deadline is $a_j + \text{max\_queue\_delay}$. On an exact tie between the
two triggers, the size trigger wins (a full batch dispatches, not a
partial one).

Batching trades throughput for latency: every request in a dispatched
batch shares the same dispatch time, so a request that arrives early in a
batch's life waits longer than one that arrives right as the batch fills.
The **added queue latency** for request $i$ is
$\text{dispatch\_time}_i - \text{arrival\_time}_i \ge 0$; averaging that
over every request measures how much delay the batching policy is
costing overall, given the observed arrival pattern.

## Task

Implement `mean_added_queue_latency(arrivals, preferred_batch_size, max_queue_delay)`:

```python
def mean_added_queue_latency(arrivals: list[int], preferred_batch_size: int, max_queue_delay: int) -> float:
    ...
```

- `arrivals`: a sequence of integer arrival timestamps (not necessarily
  pre-sorted); more than one request may share the same timestamp.
  Requests are queued FIFO.
- `preferred_batch_size`, `max_queue_delay`: positive integers, as
  defined above.

Simulate the queue event by event (each event is either "a request
arrives" or "the oldest queued item's delay cap fires") and implement
**both** triggers, with the size trigger winning ties. Return the mean,
over every request, of `dispatch_time - arrival_time`, as a Python
`float`.

## Example

```python
mean_added_queue_latency([0, 0, 3, 8, 9, 20], preferred_batch_size=3, max_queue_delay=5)
# t=0: two requests queue (delay cap for the oldest = 0+5 = 5).
# t=3: a third arrives -> size trigger -> batch of 3 dispatched at t=3.
#      latencies so far: 3, 3, 0
# t=8: one request queues alone; delay cap = 8+5 = 13.
# t=9: a second joins (size 2); nothing else arrives before t=13, so the
#      delay cap fires first -> partial batch of 2 dispatched at t=13.
#      latencies: 13-8=5, 13-9=4
# t=20: one request queues alone; delay cap = 25, nothing else arrives ->
#      dispatched at t=25. latency: 5
# mean_added_queue_latency = mean([3, 3, 0, 5, 4, 5]) == 20/6 ~= 3.333...
```

## What the gate checks

The oracle loads a committed 40-arrival stream from `arrivals.npy`
(graded with `preferred_batch_size=5, max_queue_delay=6`), plus several
seeded synthetic arrival streams with random batch sizes and delay caps,
and replays the same event-driven simulation independently — computing
every request's real dispatch timestamp, never hardcoding an expected
mean.

Your returned float is compared to the oracle's with the `rel_err`
scorer, and the worst case across every arrival stream must be `< 1e-9`.
Missing the delay trigger causes sparse tail arrivals to never flush
(simulation diverges); missing the size-trigger tie-break, or averaging
over the wrong count of requests (e.g. per-batch instead of per-request),
will land close but outside that tolerance.
