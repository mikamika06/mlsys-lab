## Context

An embedding table is a $V \times D$ matrix: $V$ vocabulary rows, each a
$D$-dimensional vector. "Gather" means fetching a handful of *whole rows*
by index — exactly what an embedding lookup does. How much memory traffic
that costs depends entirely on physical layout, not on the math:

- **Row-major**: row $v$'s $D$ elements are one contiguous run of
  $D \cdot \text{elem\_bytes}$ bytes. Gathering it touches roughly
  $\lceil D \cdot \text{elem\_bytes} / \text{line\_bytes} \rceil$ cache
  lines — a small, fixed number regardless of $V$.
- **Column-major** (the table transposed): dimension $d$'s values for
  *every* row form one contiguous run instead. A single gathered vector's
  $D$ elements are now scattered across $D$ *different* memory regions,
  each $V \cdot \text{elem\_bytes}$ bytes apart — almost every element
  needs its own fresh cache line, most of which carries 63 bytes of some
  *other* row's unrelated data along for the ride.

Same table, same gathered indices, same bytes of *useful* data returned —
wildly different memory traffic.

## Task

Implement

```cpp
void gather_line_traffic(const int* idx, int k, int V, int D, int elem_bytes,
                          int line_bytes, long* out);
```

For each of the two layouts (row-major, column-major — address formulas in
`sol.hpp`), touch every `(v, d)` pair for `v` in `idx[0..k)`, `d` in
`[0, D)`, and count the number of *distinct* `line_bytes`-byte cache lines
those addresses fall into (a line touched by more than one element still
counts once). Write the row-major count to `out[0]`, the column-major
count to `out[1]`.

## Example

$V=1000$, $D=32$, `elem_bytes=4` (so each row is exactly $32 \times 4 =
128$ bytes — precisely 2 cache lines of 64 bytes, 128-byte-aligned, so
different rows never share a line): gathering the single index $v=3$ under
row-major touches exactly 2 lines. Under column-major, that same vector's
32 elements are spread 4000 bytes apart — 32 essentially unrelated lines.

## What the gate checks

`exact_match` on `(out[0], out[1])` for a fixed gather of 5 indices
`{3, 700, 15, 999, 42}` out of a 1000x32 table. Reference:
`row_major=10, column_major=128` — better than 12x more traffic for the
identical logical gather, purely from storage layout. Swapping the two
address formulas, forgetting to de-duplicate shared lines, or iterating
`d` outside `[0, D)`, all change at least one of the two printed counts.
