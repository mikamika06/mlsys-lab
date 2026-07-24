## Context

A bump allocator hands out memory by keeping a single cursor and moving it
forward: each request rounds the cursor up to satisfy the requested
alignment, then advances past the requested size. The rounding step
introduces **padding** — bytes between the end of the previous allocation
and the start of the next one that nothing ever uses.

For a power-of-two alignment $a$, rounding an offset $x$ up to the next
multiple of $a$ is the classic bit trick:

$$
\mathrm{align\_up}(x, a) = (x + a - 1) \mathbin{\&} \sim(a - 1)
$$

The padding for that request is $\mathrm{align\_up}(x, a) - x$.

## Task

Implement

```cpp
long total_wasted_bytes(const int* sizes, const int* alignments, int n);
```

Starting from cursor offset 0, for each of the `n` requests in order:
1. Round the current offset up to a multiple of `alignments[i]` (every
   alignment is a power of two: 4, 8, 16, 32, or 64) — the gap is padding.
2. Advance the offset by that padding plus `sizes[i]`.

Return the **total padding** summed across all `n` requests — never count
the requested sizes themselves.

## Example

Starting at offset 0: request `(size=10, align=8)` needs no padding (0 is
already 8-aligned), cursor moves to 10. The next request
`(size=5, align=16)` needs padding `align_up(10, 16) - 10 = 6`, cursor
moves to `16 + 5 = 21`. Total wasted so far: `6`.

## What the gate checks

`main.cpp` builds a deterministic sequence of 30 requests with alignments
cycling through `4, 8, 16, 32, 64` and pseudo-random sizes chosen so most
requests don't already land on an aligned boundary, then prints your
returned total. The grader compiles your `.cpp` with the real local
`clang++`, runs it, and requires the printed number to match the
reference's exactly ($\mathrm{exact\_match}=1.0$). Forgetting to advance
the offset by the padding (only by `sizes[i]`) silently drifts the whole
sequence out of alignment and lands on the wrong total.
