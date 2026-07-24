## Context

**Memory-level parallelism (MLP)** is the number of independent cache misses
that can be in-flight simultaneously. A CPU's out-of-order execution engine
can overlap multiple outstanding misses, so high-MLP access patterns hide
memory latency better than low-MLP patterns.

Four archetypal traversal patterns with different MLP characteristics:

| Pattern | Description | MLP degree |
|---------|-------------|-----------|
| **Pointer chase** | Each address depends on value at previous address | 1 (serial) |
| **Sequential stream** | `a[0], a[1], ..., a[N-1]` — HW prefetcher covers it | Medium |
| **Strided (large)** | Every 16th element — multiple independent streams | Medium-high |
| **Scatter/gather** | Multiple independent random arrays accessed in parallel | High |

MLP ranking from lowest to highest: pointer_chase < sequential < strided < scatter_gather.

## Task

Implement `rank_by_mlp() -> list[str]`, which returns a list of the four
pattern names sorted from **lowest MLP** (most serial, worst latency hiding)
to **highest MLP** (most parallel, best latency hiding):

Patterns: `"pointer_chase"`, `"sequential"`, `"strided"`, `"scatter_gather"`

## Example

```python
result = rank_by_mlp()
# result == ["pointer_chase", "sequential", "strided", "scatter_gather"]
# pointer_chase is most serial (MLP=1), scatter_gather is most parallel
```

## What the gate checks

`check.py` computes the reference ranking from MLP theory and checks
`exact_match` — your returned list must exactly match the reference order.
