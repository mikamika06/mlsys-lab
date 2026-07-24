## Context

CPython uses a generational garbage collector for tracking container objects that may participate in reference cycles. Objects start in generation $0$ and can be promoted to older generations when they survive collections. When a cycle becomes unreachable, the collector can reclaim its objects.

The `gc` module exposes callbacks that receive information after each garbage collection. A callback receives the generation number and the number of objects collected during that collection. For generation $g$, the total collected objects can be measured as

$$C_g = \sum_{k \in \text{collections of generation } g} \mathrm{collected}_k.$$

This provides a direct measurement of cycle collection behavior without relying on implementation-private memory details.

## Task

Implement `objects_collected_per_generation()`:

```python
def objects_collected_per_generation() -> tuple[int, int, int]:
    ...
```

The function must run a deterministic CPython garbage collection experiment and return a tuple:

$$
(C_0, C_1, C_2)
$$

where each value is the number of unreachable objects collected by callbacks for the corresponding generation.

The experiment must:

1. Temporarily set deterministic GC thresholds.
2. Register a `gc.callbacks` listener.
3. Create unreachable reference cycles.
4. Trigger garbage collection so all three generations are observed.
5. Restore the previous GC thresholds and callback list before returning.

Return only the collected object counts for generations $0$, $1$, and $2$.

## Example

```python
result = objects_collected_per_generation()

# Example shape:
# (12, 0, 6)
```

The exact values depend on the CPython collector run, but the function must produce the same values as the reference experiment on the grading interpreter.

## What the gate checks

The gate runs a real CPython garbage collection oracle using the same public `gc` callback mechanism. It compares the returned tuple with the oracle output using exact equality.

The check verifies behavior through the runtime collector itself. It does not accept approximations based on `gc.get_stats()` counters, object counts, or hardcoded values.
