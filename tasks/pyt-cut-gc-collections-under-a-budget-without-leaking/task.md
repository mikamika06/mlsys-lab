## Context

CPython uses generational garbage collection to find unreachable reference cycles. Objects are tracked in generations, and collections happen when allocation counters exceed thresholds.

A request handler that creates many temporary cyclic objects can trigger repeated cycle scans. One optimization is to temporarily tune collection behavior while ensuring that objects are not leaked. The optimization must preserve memory safety: the number of live tracked objects after the workload should return to its starting point.

For a synthetic request loop, let $C$ be the number of garbage collection callbacks observed during the loop. The optimization target is to keep

$$C \leq B$$

for a configured collection budget $B$, while keeping the leak indicator false:

$$\mathrm{live\_after} = \mathrm{live\_before}.$$

## Task

Implement `cut_gc_collections_under_budget()`:

```python
def cut_gc_collections_under_budget() -> tuple[int, bool]:
    ...
```

The function must run a synthetic workload that creates temporary reference cycles, reduce unnecessary garbage collection activity using CPython garbage collector controls, and return:

- `collections`: the number of garbage collection events observed while the workload runs.
- `leak_flag`: `True` if the final tracked-object count does not match the initial tracked-object count, otherwise `False`.

Use CPython's `gc.freeze()` and threshold tuning appropriately. The function must restore global garbage collector state before returning.

## Example

```python
collections, leak_flag = cut_gc_collections_under_budget()

# collections is a small integer
# leak_flag is False
```

## What the gate checks

The gate runs the candidate against a CPython garbage collector oracle that performs the same workload and measures the result dynamically.

The returned tuple must exactly match the oracle result. The oracle measures collection callbacks and compares live tracked-object counts before and after the workload, so returning a fixed pair does not pass.
