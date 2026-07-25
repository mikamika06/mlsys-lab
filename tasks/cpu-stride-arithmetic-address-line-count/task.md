## Context

"Row-major is good, column-major is bad" is really a statement about
*strides*, dressed up as a statement about array layout. A 2D view is
just an addressing formula, `base + (i*row_stride + j*col_stride) *
elem_bytes` — and the same formula describes a plain row-major array, a
column-major array, or a transposed/sliced *view* of either, purely by
which of `row_stride`/`col_stride` happens to equal `1`. The real rule
isn't "prefer row-major": it's "put the stride-1 axis in the inner
loop" — whichever axis that is for the particular view you're holding.

Get the inner loop's axis right and consecutive accesses stay inside the
same cache line for a long run, each run costing one fetch no matter how
long it is. Get it backwards and (once the stride is bigger than a line)
essentially every single access opens a fresh line.

## Task

Implement two functions.

```cpp
long element_addr(long base, int i, int j, long row_stride, long col_stride, int elem_bytes);
```

Return `base + (i*row_stride + j*col_stride) * elem_bytes`.

```cpp
long traversal_fetch_count(long base, int R, int C, long row_stride, long col_stride,
                            int elem_bytes, int line_bytes, bool row_major);
```

Traverse all `R*C` elements of the view once — `row_major=true`: `i`
outer, `j` inner; `false`: `j` outer, `i` inner — tracking a single
"currently open" line (none before the first access). Each access whose
line differs from the currently-open one is a **fetch**: count it, and
that access's line becomes the new currently-open one. An access landing
in the same line as the one right before it costs nothing extra. Return
the total fetch count.

## Example

An 8x16 row-major array of 4-byte floats (`row_stride=16`,
`col_stride=1`): each row is exactly one 64-byte line
(`16 * 4 = 64`). Traversed row-major (`i` outer), every row's 16
elements stay in one line — `8` fetches total, one per row. Traversed
column-major (`j` outer, `i` inner) over that *same* array, consecutive
`i` steps jump by `row_stride * elem_bytes = 64` bytes — a brand new line
every single access — `8 * 16 = 128` fetches, 16x worse, from changing
nothing but loop order.

## What the gate checks

The driver checks `element_addr` against one address sample on each of
two views (a native `row_stride=16, col_stride=1` layout, and a
transposed `row_stride=1, col_stride=8` view), and runs
`traversal_fetch_count` on both views in both orders. It prints all six
numbers. The grader compiles `solve.cpp` with `clang++ -O2 -std=c++20`,
runs it, and requires

$$
\mathrm{exact\_match} = 1 \iff \text{all six printed numbers match the reference}
$$

The reference prints `addr_native=212 addr_transposed=344
fetch_native_rowmajor=8 fetch_native_colmajor=128
fetch_transposed_rowmajor=64 fetch_transposed_colmajor=8`. Note the
transposed view *flips* which traversal order is cheap — row-major
traversal is the bad one there (`64` fetches) and column-major is the
good one (`8`) — confirming the rule is about matching the stride-1 axis
to the inner loop, not about "row-major" as a fixed label.
