## Context

The same two-trigger dynamic batcher as before (dispatch when the queue
reaches `preferred_size`, OR when the oldest pending item has waited
`max_queue_delay` — whichever fires first, size winning ties) determines
how full the *average* batch ends up, which in turn determines
throughput.

This task ignores wall-clock time entirely: `service_time` is a fixed
model constant charged once per **formed batch**, regardless of how full
it is (the way a GPU forward pass costs roughly the same whether the
batch is 90% or 100% full). If a cap produces $M$ batches from $N$ total
requests:

$$
\overline{\text{batch\_size}} = \frac{N}{M}, \qquad
\text{throughput} = \frac{\overline{\text{batch\_size}}}{\text{service\_time}}
$$

(requests served per unit of service time). A small cap flushes partial
batches sooner, giving smaller average batches and lower throughput; a
large cap lets more requests accumulate before a delay-triggered flush,
raising both — up to the point where the size trigger already fires
first and a larger cap stops mattering.

## Task

Implement `throughput_vs_cap_curve`:

```python
def throughput_vs_cap_curve(arrivals: list[int], caps: list[int],
                             preferred_size: int, service_time: float) -> dict:
    ...
```

- `arrivals`: list of int arrival times (not necessarily sorted).
- `caps`: list of positive int candidate `max_queue_delay` values, in
  the order the output curve should preserve.
- `preferred_size`: positive int, the batcher's size trigger (fixed
  across all caps).
- `service_time`: positive float, fixed cost per formed batch.

For each cap, in the given order, simulate the two-trigger batcher over
`arrivals` and compute `mean_batch_size` and `throughput` as above.

Return `{"mean_batch_size": list[float], "throughput": list[float]}`,
each the same length and order as `caps`.

## Example

```python
arrivals = [0, 0, 1, 2, 3, 20, 21, 40, 41, 42, 60, 100, 101, 102, 103, 104, 105, 150]
caps = [3, 8, 15, 30, 50]
throughput_vs_cap_curve(arrivals, caps, preferred_size=3, service_time=2.0)
# {"mean_batch_size": [2.25, 2.25, 2.25, 2.571..., 2.571...],
#  "throughput":      [1.125, 1.125, 1.125, 1.285..., 1.285...]}
# Batch sizes plateau once the cap is large enough that some queues stop
# timing out at all before hitting preferred_size=3.
```

## What the gate checks

The grader loads a committed `arrivals.npy` fixture graded against
`caps=[3, 8, 15, 30, 50], preferred_size=3, service_time=2.0`, plus
several additional seeded random scenarios, and computes both curves
with an independent implementation of the same batcher simulation and
formulas — never calling your function, never hardcoding an expected
curve.

`rel_err` is `scorers.rel_err` applied to the concatenation of both
curves against the oracle's, taking the worst case across all scenarios,
and must be `<= 1e-9`. An off-by-one in when a batch is counted, mixing
up which trigger wins a tie, or dividing by the wrong count all shift
the entire curve.
