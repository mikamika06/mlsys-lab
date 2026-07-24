## Context

Modern CPUs contain a **hardware prefetcher** that monitors memory access
patterns and speculatively loads data before it is needed. Most hardware
prefetchers implement one or both of:

- **Stream prefetcher**: detects sequential or fixed-stride access patterns and
  issues prefetch requests ahead of time.
- **Stride prefetcher**: detects a constant stride $s$ between consecutive
  accesses to the same load instruction.

A pattern is **caught** by the prefetcher if it is regular enough (sequential
or constant stride) that the prefetcher can predict future accesses. Irregular
or pointer-chasing patterns defeat the prefetcher.

Specifically:
- **Sequential** access (stride = element size): caught ✓
- **Fixed stride** (e.g., every 2nd, 4th element): caught ✓
- **Random** (uniformly shuffled indices): not caught ✗
- **Pointer chasing** (linked list, each access depends on previous value): not
  caught ✗ — the next address is unknown until the load completes
- **Strided with very large stride** ($\ge$ page size = 4096 B): generally not
  caught ✗ (hardware prefetchers stop tracking at page boundaries)

## Task

Implement `classify_prefetch() -> list[bool]`, which returns a list of 5
booleans. Element $k$ is `True` if pattern $k$ is caught by a typical
stream+stride hardware prefetcher, `False` otherwise.

The five patterns (each over 256 elements, element size = 4 bytes / `int32`):

| # | Pattern | Description |
|---|---------|-------------|
| 0 | Sequential: `a[0], a[1], ..., a[255]` | stride = 4 B |
| 1 | Fixed stride: `a[0], a[4], a[8], ..., a[1020]` | stride = 16 B |
| 2 | Random: shuffled indices | no predictable stride |
| 3 | Pointer-chase: next address from current data | future addr unknown until load |
| 4 | Large stride: `a[0], a[1024], a[2048], ...` | stride = 4096 B (one page) |

## Example

```python
result = classify_prefetch()
# result[0] == True   (sequential — classic stream prefetch)
# result[1] == True   (stride 16B — stride prefetch catches it)
# result[2] == False  (random — defeats prefetcher)
# result[3] == False  (pointer-chase — each addr depends on prior load)
# result[4] == False  (stride >= page size, crosses page boundary each step)
```

## What the gate checks

`check.py` computes the reference labels from the prefetcher rules above and
checks `exact_match` — your returned list must agree on all five patterns.
