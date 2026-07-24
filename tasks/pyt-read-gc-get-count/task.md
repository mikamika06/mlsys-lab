## Context

CPython uses a generational garbage collector to track container objects that may
participate in reference cycles. The collector maintains counters for its
generations. The function `gc.get_count()` exposes these counters as a tuple

$$
(c_0, c_1, c_2),
$$

where each value represents the current allocation count tracked for one GC
generation.

Calling `gc.collect()` performs a collection pass and resets the tracked
generation counters according to CPython's internal GC state. The exact values
are implementation details, so this task uses a controlled CPython 3.12
environment and requires reading the real runtime state instead of predicting
fixed numbers.

## Task

Implement `measure_gc_count(n)`.

The function must:

1. Temporarily disable automatic garbage collection with `gc.disable()`.
2. Allocate `n` temporary empty list objects.
3. Remove the temporary references.
4. Run a manual full collection with `gc.collect()`.
5. Return the tuple returned by `gc.get_count()` after the collection.
6. Restore the previous GC enabled/disabled state before returning.

The return value must be the three-element tuple produced by CPython.

Example:

```python
import gc

count = measure_gc_count(1000)
# count is a tuple such as (0, 0, 0) in CPython 3.12
```

## What the gate checks

The gate runs a real CPython garbage collector oracle. It performs the same
controlled allocation sequence with automatic GC disabled, calls the real
`gc.collect()`, and reads the real `gc.get_count()` result.

The returned tuple from `measure_gc_count` must exactly match the oracle tuple.
Implementations that return `gc.get_threshold()`, guess fixed values, or skip the
manual collection will fail.
