## Context

Modern compilers (GCC, Clang) automatically vectorize loops using SIMD
instructions (AVX on x86, NEON on ARM/Apple Silicon) when they can *prove* that:

1. **No loop-carried dependencies** — iteration $i$ must not read a value
   written by a prior iteration $i' < i$.
2. **No pointer aliasing** — arrays accessed in the loop do not overlap in
   memory (or the compiler can prove so statically).
3. **Straightforward reduction** — reductions ($\sum$, $\max$) must use
   patterns the compiler recognizes (e.g. plain `+=` on a scalar accumulator).
4. **Uniform stride** — the access pattern must be a known stride (typically
   stride-1 for SIMD loads).

A loop is **vectorizable** if *all* of the above conditions are met.

## Task

Implement `classify_loops() -> list[bool]`, which returns a list of five
booleans. Element $k$ is `True` if loop $k$ can be auto-vectorized (no
dependency/aliasing/reduction issue), and `False` otherwise.

The five loops are described below (Python pseudocode for illustration; assume
`a`, `b`, `c` are distinct, non-overlapping arrays):

| # | Loop body | Vectorizable? |
|---|-----------|--------------|
| 0 | `a[i] = b[i] + c[i]` | — element-wise, no dep |
| 1 | `a[i] = a[i-1] + b[i]` | — loop-carried dep on `a` |
| 2 | `s += a[i]` | — simple reduction, vectorizable |
| 3 | `a[i] = b[i] if b[i] > 0 else 0` | — conditional, branch-free SIMD select |
| 4 | `a[i] = b[i * i % N]` | — non-uniform (quadratic) index stride |

## Example

```python
result = classify_loops()
# result[0] == True    (simple element-wise add)
# result[1] == False   (loop-carried dependency)
# result[2] == True    (recognized reduction)
# result[3] == True    (branch-free select / max(0, x))
# result[4] == False   (non-uniform stride prevents vectorization)
```

## What the gate checks

`check.py` computes the reference classification from the static dependency
rules above and checks `exact_match` — your returned list must agree on all
five loops.
