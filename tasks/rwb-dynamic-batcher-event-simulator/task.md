## Context

Inference servers such as Triton use a **dynamic batcher**: incoming
requests queue up and are dispatched together as soon as either of two
independent triggers fires, whichever happens first:

- **Size trigger.** The queue reaches `preferred_batch_size`: dispatch a
  full batch of exactly `preferred_batch_size` items immediately.
- **Delay trigger.** The oldest item still waiting in the queue has been
  there for `max_queue_delay` time units: dispatch the **entire current
  queue** right away, even if it is smaller than `preferred_batch_size`
  (a *partial* batch). Without this trigger, a queue that never quite
  fills up would simply never be flushed.

If $a_j$ is the arrival time of the oldest item currently queued, its
delay-trigger deadline is $a_j + \text{max\_queue\_delay}$. If the size
trigger and the delay trigger would fire at the exact same instant, the
size trigger wins: a full batch is what dispatches at that timestamp, not
a partial one.

## Task

Implement `dynamic_batcher_simulate(arrivals, preferred_batch_size, max_queue_delay)`:

```python
def dynamic_batcher_simulate(arrivals, preferred_batch_size, max_queue_delay):
    ...
```

- `arrivals`: a sorted (non-decreasing) sequence of integer arrival
  timestamps; more than one request may share the same timestamp.
  Requests are queued FIFO.
- `preferred_batch_size`, `max_queue_delay`: positive integers, as
  defined above.

Simulate the queue event by event (each event is either "a request
arrives" or "the oldest queued item's delay cap fires") and implement
**both** triggers. Return a pair `(batch_formation_times, batch_sizes)` of
equal-length sequences, in the order batches are formed:

- `batch_formation_times[i]`: the timestamp batch $i$ was dispatched at.
- `batch_sizes[i]`: how many requests were in batch $i$ (`<=
  preferred_batch_size`).

Every arrival must end up in exactly one batch, so
`sum(batch_sizes) == len(arrivals)`.

## Example

```python
times, sizes = dynamic_batcher_simulate([0, 0, 3, 8, 9, 20], preferred_batch_size=3, max_queue_delay=5)
# times == [3, 13, 25]
# sizes == [3, 2, 1]
#
# t=0: two requests queue up (size 2, delay cap for the oldest is 0+5=5).
# t=3: a third arrives -> size trigger -> batch (3, 3) dispatched.
# t=8: one request queues alone; delay cap = 8+5 = 13.
# t=9: a second joins it (size 2); nothing else arrives before t=13, so
#      the delay cap fires first -> partial batch (13, 2).
# t=20: one request queues alone; delay cap = 20+5 = 25, nothing else
#      arrives -> partial batch (25, 1).
```

## What the gate checks

The gate loads a committed 40-arrival stream from `arrivals.npy` (graded
with `preferred_batch_size=5, max_queue_delay=6`), plus several additional
seeded random arrival streams with random `preferred_batch_size` and
`max_queue_delay`, and replays the same event-driven simulation
independently in Python for each — never calling your function, never
hardcoding an expected batch list. Your returned `(batch_formation_times,
batch_sizes)` must match the oracle's, entry-for-entry, on every case
(`exact_match == 1.0`). Missing the delay trigger causes trailing (or
sparse) runs of arrivals to accumulate forever instead of flushing as a
partial batch; missing the size-trigger tie-break causes an off-by-one in
both the timestamp and size of whichever batch straddles a tie.
