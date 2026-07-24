## Context

Given the same trace of `N` requests (fixed arrival order, generation
length $\ell_i$ each) and the same number of concurrent slots
`batch_size`, two batching policies finish the trace at very different
wall-clock times:

- **Static batching**: requests are grouped, in arrival order, into
  consecutive batches of `batch_size` (the last batch may be smaller).
  Batches run strictly one after another — batch $b+1$ cannot start until
  every request in batch $b$ has finished — and every slot in a batch is
  held **for the whole batch's duration**, even if its own request
  finished early: batch $b$'s duration is $\max(\ell_i : i \in b)$.
  Total time to clear the trace:
  $$
  \text{makespan}_{\text{static}} = \sum_{b} \max_{i \in b} \ell_i
  $$

- **Continuous batching**: `batch_size` slots run work-conserving —
  whenever a slot frees, it immediately takes the next waiting request in
  arrival order, with no batch boundaries. The trace is cleared once the
  last request finishes:
  $$
  \text{makespan}_{\text{cont}} = \max_i (\text{completion time of request } i)
  $$
  (simulated exactly like list scheduling: each slot tracks its own free
  time, initially $0$; each arriving request goes to whichever slot is
  free soonest, and that slot's free time becomes its completion time.)

Continuous batching can only finish sooner or at the same time as static
batching for the same trace and slot count — it never holds a slot idle
waiting for a batch-mate — so its throughput is always at least as good.

## Task

Implement `compare_batching_throughput`:

```python
def compare_batching_throughput(
    gen_lens: list[int], batch_size: int,
) -> tuple[float, float, float]:
    ...
```

- `gen_lens`: a list of `N` positive ints, the generation length of each
  request **in arrival order** (both policies process this exact order).
- `batch_size`: number of concurrent slots (also the static batch size).

Return `(makespan_static, makespan_cont, throughput_ratio)`:

- `makespan_static` — as defined above.
- `makespan_cont` — as defined above.
- `throughput_ratio` — $\dfrac{\text{tokens}/\text{makespan}_{\text{cont}}}{\text{tokens}/\text{makespan}_{\text{static}}}$,
  where `tokens = sum(gen_lens)` (the same total for both policies, so
  this simplifies to $\text{makespan}_{\text{static}} / \text{makespan}_{\text{cont}}$,
  always $\ge 1$: continuous batching's speedup factor over static on
  this trace).

## Example

```python
gen_lens = [1, 1, 1, 10]
batch_size = 2

compare_batching_throughput(gen_lens, batch_size)
# static: batch 0 = [1,1] -> duration 1; batch 1 = [1,10] -> duration 10
#   makespan_static = 1 + 10 = 11
# continuous: slot A: 1 -> 1, then 1 -> 2, then 10 -> 12 ... actually the
#   greedy always fills the free-soonest slot, so it interleaves work
#   more tightly than the batch boundary allows; simulate it exactly as
#   described to get makespan_cont.
# throughput_ratio = makespan_static / makespan_cont >= 1
```

## What the gate checks

The grader builds several `(gen_lens, batch_size)` scenarios from a
seeded NumPy generator (skewed traces with one long request among many
short ones, `batch_size = 1`, `batch_size = N`, and uneven trace lengths
not divisible by `batch_size`) and computes `makespan_static`,
`makespan_cont`, and the ratio independently in NumPy, following the
definitions above exactly, never calling your function or hardcoding an
expected value.

`rel_err` is the worst-case relative error across **all three** returned
scalars, over every scenario, and the gate requires `<= 1e-9`. Treating
static batches as if they could pipeline (starting the next batch before
the previous one's longest request finishes), using `sum` instead of
`max` within a static batch, assigning continuous-batching arrivals to
slots in the wrong order, or getting the ratio upside down will all
produce a visible mismatch on at least one scenario — the sanity checks
(`batch_size = 1` and `batch_size = N`, where both policies coincide and
`throughput_ratio` must land at exactly `1.0`) are included specifically
to catch an inverted ratio.
