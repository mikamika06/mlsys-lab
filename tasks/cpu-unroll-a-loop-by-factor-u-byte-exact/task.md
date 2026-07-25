## Context

**Loop unrolling** replaces a loop that does one unit of work per
iteration with one that does several ($U$) units per iteration,
reducing per-iteration overhead (loop-counter increment, bound check,
branch) and giving the compiler more independent instructions to
schedule. Unrolling by factor $U$ means the loop now advances $U$
elements at a time: for outer step $b \in \{0, U, 2U, \dots\}$, it
processes every element $b+k$ for $k \in [0, U)$ before moving to the
next block.

Unrolling is a pure restructuring of *how* the work is scheduled, not
*what* is computed -- for an elementwise operation like
$y_i \mathrel{+}= a \cdot x_i$, where each $i$'s result depends on
nothing but $x_i$ and $y_i$, the output must be bit-for-bit identical
regardless of $U$. Getting the block-index arithmetic wrong (e.g. an
off-by-one that skips or repeats an element between blocks) is a
classic real-world unrolling bug, and it usually only shows up for
specific values of $U$ -- which is exactly why this task checks several.

## Task

Implement:

```cpp
void axpy_unrolled(int n, int U, float a, const float* x, float* y);
```

For every $i \in [0, n)$, compute `y[i] = y[i] + a * x[i]` in place.
`n` is always an exact multiple of `U`. Structure the loop as an outer
loop over blocks of `U` elements (`for (int b = 0; b < n; b += U)`),
processing all `U` elements of each block -- element `k` of block `b`
is at index `b + k`, for `k` in `[0, U)` -- before advancing.

## Example

For `n = 4`, `U = 2`, `a = 1.7`, `x = [1, 2, 3, 4]`, `y = [0, 0, 0, 0]`:
block `b = 0` processes indices `0` and `1`
(`y = [1.7, 3.4, 0, 0]`), block `b = 2` processes indices `2` and `3`
(`y = [1.7, 3.4, 5.1, 6.8]`).

## What the gate checks

`main.cpp` fixes one 12-element `(x, y_init, a)` triple and runs
`axpy_unrolled` at every divisor of `12` (`U = 1, 2, 3, 4, 6, 12`),
resetting `y` from `y_init` before each trial, printing the full
resulting `y` array every time. The candidate's full stdout is compared
byte-for-byte (`exact_match = 1.0`) against the reference's, whose six
printed rows are all identical to each other -- the choice of `U`
changes nothing about the answer. An unimplemented function leaves `y`
untouched, printing `y_init` unchanged instead of the correct
`y_init + a * x`, and fails on every row.
