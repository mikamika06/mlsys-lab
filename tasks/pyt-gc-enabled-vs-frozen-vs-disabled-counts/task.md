## Context

CPython uses a generational garbage collector to find unreachable reference cycles.
The collector maintains statistics for each generation. For generation $i$, the
number of automatic collections can be read from:

$$
\mathrm{collections}_i = \mathrm{gc.get\_stats()}[i]["collections"] .
$$

The collector can be controlled through the `gc` module. When enabled, automatic
collections may happen while objects are allocated. When disabled, automatic
collections are skipped. The `gc.freeze()` operation moves currently tracked
objects into a permanent generation so they are ignored by future collection
passes.

A useful experiment is to perform the same allocation workload under several GC
states and compare the observed collection counts. The workload must be identical
for each state so that only the collector mode changes the result.

## Task

Implement `gc_collection_counts()`.

The function must return a tuple:

```python
(enabled_count, frozen_count, disabled_count)
```

Each value is the number of automatic garbage-collector collections that happen
during the same allocation workload.

Use the following procedure:

1. Save the current GC state and thresholds.
2. For each mode, reset the GC statistics baseline, apply the mode, and run the
   allocation workload.
3. Return the increase in the total collection count across all generations.

The allocation workload should create many short-lived cyclic objects so that
cycle collection behavior is exercised. The three modes are:

- `enabled`: normal automatic GC enabled.
- `frozen`: call `gc.freeze()` before running the workload, with automatic GC
  enabled.
- `disabled`: automatic GC disabled.

Restore GC state before returning.

## Example

```python
result = gc_collection_counts()

# Example shape only; exact values depend on CPython.
# (2, 2, 0)
assert len(result) == 3
assert all(isinstance(x, int) for x in result)
```

## What the gate checks

The gate runs a CPython garbage-collector oracle that performs the same workload
and measures collection counter deltas using `gc.get_stats()`. Your function must
return exactly the same three counts as the oracle.
