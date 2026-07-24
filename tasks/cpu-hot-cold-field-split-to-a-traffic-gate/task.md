## Context

An array of structures (AoS) stores every record's fields together. This is convenient for general access, but a loop that repeatedly reads only a few hot fields may still fetch unused cold bytes into the cache.

A structure with total size $S$ bytes occupies cache lines according to the accessed byte addresses. A cache line size of $L$ bytes means an address $x$ maps to line

$$
\left\lfloor \frac{x}{L} \right\rfloor .
$$

A structure-of-arrays (SoA) layout separates frequently accessed fields from rarely accessed fields. A hot loop can then walk a compact hot array, reducing the number of cache lines loaded.

The task models a deterministic cache. The returned address trace is passed through a simulator with fixed parameters. The goal is to produce a trace representing a cache-friendly hot/cold split rather than an AoS layout that mixes unrelated bytes.

## Task

Implement `hot_cold_trace(n)`:

```python
def hot_cold_trace(n: int) -> list[int]:
    ...
```

Return a list of byte addresses accessed by a loop over `n` records.

Assume the original AoS layout has records of size $32$ bytes. Each record contains a hot field at offset $0$ and cold fields occupying the remaining bytes.

Design the access trace for a hot loop that needs only the hot field from every record. The returned addresses should model a split layout where hot values are stored contiguously. The function must not use randomness.

## Example

```python
trace = hot_cold_trace(4)
# A possible SoA-style trace:
# [0, 4, 8, 12]
```

The exact addresses should follow a deterministic contiguous hot-array layout.

## What the gate checks

The gate runs the returned byte-address trace through a deterministic cache simulator. The simulator uses fixed line size, set count, and associativity parameters, so the result does not depend on the machine running the task.

The metric is `modeled_cache_misses`. A solution passes when its simulated cache traffic stays below the configured miss threshold. A trace that repeatedly touches full AoS records instead of a compact hot array will load more cache lines and fail the gate.
