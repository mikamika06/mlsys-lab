## Context

A dynamic (request-level) batcher queues incoming requests and dispatches
them as a batch under two independent triggers, whichever fires first:

- **Size trigger** — the pending queue reaches `preferred_size`: dispatch
  immediately, a full batch of exactly `preferred_size` items.
- **Delay trigger** — the oldest item still in the queue has been
  waiting `max_queue_delay` time units: dispatch the **entire current
  queue**, even if it has fewer than `preferred_size` items (a
  **partial** batch). Without this trigger, a queue that never quite
  fills up would simply never be dispatched.

If arrival $a_j$ is the oldest pending item, its delay-trigger deadline
is $a_j + \text{max\_queue\_delay}$. At a tie between the two triggers at
the same instant, the size trigger wins (an arrival that completes a
full batch dispatches immediately, even if the delay cap for the oldest
item fires at that exact same time).

## Task

`tasks/rwb-debug-a-batcher-that-ignores-the-delay-cap/starter.py` contains
a broken `batch_formation_times` that only implements the size trigger —
the delay trigger is missing entirely, so any trailing (or sparse) run of
arrivals that never reaches `preferred_size` is silently dropped and
never dispatched. Fix it:

```python
def batch_formation_times(arrivals: list[int], preferred_size: int, max_queue_delay: int) -> list[tuple[int, int]]:
    ...
```

- `arrivals`: sorted (non-decreasing) list of integer arrival times; more
  than one request may share the same arrival time. Requests are queued
  FIFO.
- `preferred_size`, `max_queue_delay`: positive ints, as defined above.
- Simulate the queue over time and implement BOTH triggers.
- Return a list of `(formation_time, batch_size)` tuples, in the order
  batches are formed. Every arrival must end up in exactly one batch,
  and every batch size must be `<= preferred_size`.

## Example

```python
arrivals = [0, 0, 0, 5, 9, 20, 21, 22, 23, 50, 51, 90]
batch_formation_times(arrivals, preferred_size=4, max_queue_delay=10)
# [(5, 4), (19, 1), (23, 4), (60, 2), (100, 1)]
#
# t=0: three requests queue up (size 3, no trigger).
# t=5: a 4th arrives -> size trigger -> batch (5, 4) dispatched.
# t=9: one request queues alone; no more arrivals until t=20, so its
#      delay cap (9 + 10 = 19) fires FIRST -> partial batch (19, 1).
# t=20..23: four requests queue up one at a time; the 4th (t=23) hits
#      the size trigger -> batch (23, 4).
# t=50, 51: two requests queue; nothing else arrives before their delay
#      cap (50 + 10 = 60) fires -> partial batch (60, 2).
# t=90: one request queues alone; delay cap (90 + 10 = 100) fires ->
#      partial batch (100, 1).
```

## What the gate checks

The grader loads a committed `arrivals.npy` fixture (the exact stream
above) graded with `preferred_size=4, max_queue_delay=10`, plus several
additional seeded random arrival streams with random `preferred_size`
and `max_queue_delay`, and replays the same event-driven simulation
independently in Python — never calling your function, never hardcoding
an expected batch list.

`exact_match` is the fraction of oracle-produced `(formation_time,
batch_size)` entries, across all cases, that your output matches
position-for-position. The gate requires `1.0`. The buggy starter never
checks the delay cap, so on the fixture stream it keeps accumulating
past where the correct simulation would have flushed early: it produces
`[(5, 4), (22, 4), (90, 4)]` instead of the correct `[(5, 4), (19, 1),
(23, 4), (60, 2), (100, 1)]` — only the very first batch lines up, and
every batch after it forms at the wrong time, with the wrong size, or
not at all.
