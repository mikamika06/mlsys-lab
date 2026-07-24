## Context

A **stride-`s` walk** reads `n` elements at indices `0, s, 2s, ..., (n-1)s`
instead of `0, 1, 2, ..., n-1`. Each element is `elem_bytes` bytes, and a
cache line holds `L = line_bytes / elem_bytes` elements. How many
*distinct* cache lines does the walk touch?

You could find out by simulating: walk all `n` indices, compute each
one's line number, and count the distinct values. But the answer has a
closed form, and understanding *why* is the point of spatial locality —
it's the difference between "measure it" and "know it in advance."

The line index of access `i` is `floor(i * s / L)`. As `i` increases by 1,
that value increases by `s / L`, rounded down to the next integer
boundary:

- If `s <= L` (stride fits inside a line, or exactly spans one), the
  per-step increase is `<= 1`, so consecutive accesses land on the same
  line or the very next one — the lines touched form one **contiguous
  run** from line `0` to line `floor((n-1) * s / L)`, with no gaps.
- If `s > L` (stride overshoots a line), the per-step increase is `> 1`
  wherever it doesn't wrap into some coincidental interaction, so the
  line sequence is **strictly increasing** — every access lands on a line
  none of the previous accesses touched, giving exactly `n` distinct
  lines.

## Task

Implement

```cpp
long distinct_lines_stride_walk(long n, long stride, long elem_bytes, long line_bytes);
```

which returns the number of distinct `line_bytes`-byte lines touched by
reading `n` elements of `elem_bytes` bytes each at byte addresses
`0, stride*elem_bytes, 2*stride*elem_bytes, ..., (n-1)*stride*elem_bytes`.
`line_bytes` is always an exact multiple of `elem_bytes`. Derive the
result algebraically — do not simulate the walk element by element.

## Example

$$
n = 100,\ \text{stride} = 17,\ \text{elem\_bytes} = 4,\ \text{line\_bytes} = 64
$$

Here `L = 64/4 = 16` and `stride = 17 > L`, so every one of the 100
accesses lands on a line none of the others do: the answer is `100`. With
`stride = 2` instead (`<= L`), the accesses cluster into one contiguous
run of lines `0` through `floor(99*2/16) = 12`, i.e. `13` distinct lines.

## What the gate checks

The driver runs 20 fixed `(n, stride, elem_bytes, line_bytes)` cases,
spanning strides below, equal to, and above `L`, at two element sizes and
two line sizes. For each case it computes ground truth by actually
walking all `n` accesses into a `std::set` of line indices (this is the
"simulator" your closed form is checked against, not a hardcoded value),
compares that to your function's output, and prints how many of the 20
cases agree. The grader compiles `solve.cpp` with `clang++ -O2
-std=c++20`, runs it, and requires

$$
\mathrm{exact\_match} = 1 \iff \text{printed agreement count matches the reference}
$$

The reference agrees on all 20 (`agree=20/20`). A formula that only
handles one regime — e.g. always returning `n`, which is correct only
when `stride >= L` — disagrees on every small-stride case where lines
should merge into a contiguous run, so `agree` comes out below 20 and the
gate fails.
