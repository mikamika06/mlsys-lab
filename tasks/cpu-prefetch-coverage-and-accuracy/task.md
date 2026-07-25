## Context

Two prefetch requests can both be "correct" in the sense of not being
wrong, yet a prefetcher can still be bad in two different ways:

- **Coverage**: of all the misses that would have happened with no
  prefetching, how many did the prefetcher actually eliminate?
- **Accuracy**: of all the prefetches it issued, how many were actually
  used before being evicted (versus wasted bandwidth and cache space on
  a line nothing ever touched)?

A prefetcher can have perfect accuracy but terrible coverage (it only
ever prefetches a few, always-correct, addresses) or perfect coverage
but terrible accuracy (it prefetches everything in sight, most of it
useless).

## Task

Implement

```cpp
void compute_coverage_accuracy(long baseline_misses, long total_prefetches, long useful_prefetches,
                                double* coverage_out, double* accuracy_out);
```

$$\text{coverage} = \frac{\text{useful\_prefetches}}{\text{baseline\_misses}}, \qquad \text{accuracy} = \frac{\text{useful\_prefetches}}{\text{total\_prefetches}}$$

## Example

The driver runs a fixed 26-access trace (16 sequential lines, then 10
lines strided 2 apart) once with no prefetching and once with a
next-line prefetcher. The sequential run is the prefetcher's best case —
every prefetch it issues there gets consumed. The strided run is its
worst case — every prefetch targets the very next line, but the trace
always skips one further, so none of those prefetches are ever used.
Overall this trace gives `baseline_misses = 26`, `total_prefetches = 18`,
`useful_prefetches = 8`, so `coverage = 8/26 ≈ 0.3077` and
`accuracy = 8/18 ≈ 0.4444` — a prefetcher that is right less than half
the time it fires, and only eliminates about a third of the misses it
could have.

## What the gate checks

`max_abs_err` on the two printed ratios (the 3 input counts come from a
fixed harness-owned simulation and are identical for every candidate).
Swapping the two denominators, or computing a difference instead of a
ratio, gives values nowhere near `0.3077` / `0.4444`; a starter
returning `0.0, 0.0` fails outright.
