## Context

GPipe splits a model into $p$ sequential stages, one per device, and runs a
mini-batch as $m$ microbatches pushed through the pipeline. Each device
alternates between working on a microbatch and sitting idle waiting for
upstream/downstream data. For a single pipeline "flush" (all $m$ microbatches'
forward passes, then all $m$ backward passes, GPipe's simple schedule), a
device's timeline breaks into fixed-size slots (each slot = the time to run
one forward or backward micro-step). Two things happen in those slots across
the whole flush:

* **Active slots** — every device does exactly $2m$ units of real work
  (one forward + one backward per microbatch)... but the pipeline *fill* and
  *drain* mean devices can't all be busy at once. The standard bubble-time
  derivation counts it in units where one "slot" is the time for a device to
  process one microbatch's forward-or-backward step, and the fill/drain
  overhead is $p - 1$ such slots' worth of *idle* time per device, on top of
  $m$ slots' worth of pipeline "throughput" time (this is the textbook GPipe
  bubble analysis, e.g. Huang et al. 2019).
* **Idle ("bubble") slots** — $p - 1$, the pipeline fill+drain overhead that
  doesn't depend on $m$.

## Task

Implement:

```python
def gpipe_bubble_fraction(microbatches: int, stages: int) -> float:
    ...
```

* `microbatches` — number of microbatches per flush, $m \ge 1$.
* `stages` — number of pipeline stages, $p \ge 1$.

Return the bubble fraction

$$
\text{bubble}(m, p) = \frac{p - 1}{m + p - 1}.
$$

This is the fraction of a device's time spent idle rather than doing useful
forward/backward work, under GPipe's schedule. As $m \to \infty$ (many
microbatches per flush) the bubble fraction shrinks toward 0; as $m \to 1$
it grows toward $(p-1)/p$.

## Example

```python
gpipe_bubble_fraction(microbatches=8, stages=4)
# (4 - 1) / (8 + 4 - 1) = 3 / 11 = 0.2727...

gpipe_bubble_fraction(microbatches=1, stages=1)
# (1 - 1) / (1 + 1 - 1) = 0.0   (a single stage never idles on itself)
```

## What the gate checks

A single gate, **rel_err**, compares your return value against
$(p-1)/(m+p-1)$ computed directly, over a range of `(microbatches, stages)`
pairs including edge cases ($p=1$, $m=1$, large $m$, large $p$). Relative
error must be `< 1e-9`.
