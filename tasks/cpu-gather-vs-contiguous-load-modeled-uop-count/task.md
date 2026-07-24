## Context

Modern CPUs execute SIMD vector loads and scalar (gather) loads with different
micro-operation (uop) counts.  A **contiguous aligned** vector load of $vw$
elements costs **1 load uop**.  A **scatter / gather** load that fetches $m$
non-contiguous elements costs **$m$ load uops** (one per element) because each
address must be calculated and issued separately.

A 64-byte **cache line** holds $64 / e$ elements of element size $e$ bytes.
Sequential accesses that stay within one line cause only 1 compulsory miss;
jumping to a new line (as a gather pattern does when the access order is
scrambled) incurs additional misses.

## Task

Implement `modeled_load_uops(m, vw, e)` that returns a dictionary with four
keys:

| Key | Meaning |
|-----|---------|
| `contiguous_uops` | Modeled load-uop count for a **vectorised contiguous** load of $m$ elements with vector width $vw$: $\lceil m / vw \rceil$ |
| `gather_uops` | Modeled load-uop count for a **scalar gather** of $m$ elements: $m$ |
| `contiguous_misses` | Cache misses (via the simulator) for a **sequential** access of $m$ contiguous elements, each $e$ bytes |
| `gather_misses` | Cache misses for a **scrambled-order** access of the **same** $m$ elements |

Cache parameters are fixed: `line_bytes=64, sets=64, ways=8`.
Element $i$ maps to byte address $i \times e$.

Use Python's `random` module with seed **42** to produce the scrambled order
(deterministic across runs).

## Example

```python
result = modeled_load_uops(100, 4, 8)
# result["contiguous_uops"] == 25        # ceil(100/4)
# result["gather_uops"]      == 100      # one uop per element
```

## What the gate checks

`check.py` computes the reference answer using the same algorithm (uop formula
plus `arena.cachesim.simulate` for miss counts) and compares the learner's
result element-by-element.  Each gate returns 1 (pass) or 0 (fail):

| Metric | Condition | What it verifies |
|--------|-----------|------------------|
| `contiguous_uops` | exact match | $\lceil m / vw \rceil$ formula |
| `gather_uops` | exact match | $m$ uops for scatter |
| `contiguous_misses` | exact match | sequential trace miss count |
| `gather_misses` | exact match | scrambled trace miss count |

All four must be 1 for the task to pass.
