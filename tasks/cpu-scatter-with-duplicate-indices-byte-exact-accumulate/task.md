## Context

Scatter operations copy data from a source array into a destination array at positions given by an index array. When indices repeat, the result depends on the accumulation policy: does the second write overwrite the first, or do the values accumulate? In Python, `reference_add_at(dst, idx, src)` performs atomic accumulation — each source element is *added* to the destination at the given index, so duplicate indices sum their contributions.

In a cache-coherent CPU, multiple writes to the same byte address create a **write-after-write** hazard. A naive implementation that writes sequentially will end up overwriting instead of accumulating, producing a wrong result.

This task fixes a broken scatter implementation that fails on duplicate indices.

## Task

Fix the function `scatter_add(dst, idx, src, out)` so that it performs **byte-exact accumulate**: for each `i`, `out[idx[i]] += src[i]`. The arrays are 1-D `int32` list. All are writable except `src`. `out` is pre-allocated and initially a copy of `dst`. The function returns `None` (modifies `out` in place).

Your fix must handle duplicate indices correctly: all contributions must sum, not overwrite.

## Example

```python
dst = [1, 2, 3, 4]
idx = [0, 1, 0, 2]
src = [10, 20, 30, 40]
out = dst.copy()
scatter_add(dst, idx, src, out)
print(out)  # [1+10+30, 2+20, 3+40, 4] = [41, 22, 43, 4]
```

## What the gate checks

The grader runs your `scatter_add` on a random test case with duplicate indices, compares `out` byte-exact against `reference_add_at(dst, idx, src)`. Gate `covers_all` checks no index out of bounds; gate `byte_exact` checks every byte matches.
