## Context

A row-major $N \times N$ matrix of 8-byte elements stores element
$(row, col)$ at byte address

$$
\mathrm{addr}(row, col) = (row \cdot N + col) \cdot 8.
$$

A cache line holds several consecutive bytes -- at 64 bytes per line and
8 bytes per element, one line holds 8 consecutive elements of the *same
row*. Sweeping every element of the matrix exactly once can be written
two ways:

- **column-major** (`for (col) for (row)`): the inner loop varies
  `row`, so consecutive touches jump `N * 8` bytes apart -- for any
  `N` large enough that a line only holds a handful of elements, every
  single touch lands in a different line. None of a line's other 7
  elements get visited again until the traversal comes back around to
  that column much later, by which point the line has long since been
  evicted.
- **row-major** (`for (row) for (col)`): the inner loop varies `col`,
  so consecutive touches are 8 bytes apart -- 8 touches in a row land in
  the *same* 64-byte line before moving to the next one.

**Loop interchange** is exactly this: swapping the nesting order of two
loops so the innermost one walks the array's physically contiguous
(stride-1) dimension. It changes *only the order* elements are visited
in, not which elements or how many -- but that's enough to turn "1
useful byte per line fetched" into "8 useful bytes per line fetched".

## Task

Implement:

```cpp
void row_major_traverse(int N);
```

Using the `touch(byte_addr)` hook and the `elem_addr(N, row, col)`
helper (both declared in `sol.hpp`), visit every one of the matrix's
`N * N` elements **exactly once**, in row-major order: for each `row`
from `0` to `N-1`, walk every `col` from `0` to `N-1` in that row before
moving to the next row.

## Example

For `N = 4`, `row_major_traverse` must call `touch` in exactly this
address order (each `elem_addr(4, row, col)`, 8 bytes apart within a
row): `(0,0), (0,1), (0,2), (0,3), (1,0), (1,1), (1,2), (1,3), (2,0),
...` -- 16 touches total, one per element, row by row.

## What the gate checks

`main.cpp` sweeps a 64x64 matrix of doubles (32768 bytes, four times the
harness's 8192-byte modeled cache) through a fixed **column-major**
baseline traversal (the harness's own code, not the learner's) on one
fresh cache, then through the candidate's `row_major_traverse` on a
second fresh cache, and prints both miss counts. The reference's
row-major order gets `512` misses (one compulsory miss per 64-byte line,
`32768 / 64`, since every group of 8 consecutive elements it visits
shares a line); the column-major baseline gets `4096` -- one miss per
element, since it never revisits a line before moving on. The
candidate's full stdout is compared byte-for-byte (`exact_match = 1.0`)
against the reference's, so an empty or column-major-equivalent
implementation prints the wrong `row_major_misses` (`0` or `4096`
instead of `512`) and fails.
