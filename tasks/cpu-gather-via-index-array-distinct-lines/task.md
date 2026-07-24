## Context

A gather reads `base[idx[i]]` for a list of indices `idx` instead of
walking `base` sequentially. The CPU still fetches memory 64 bytes at a
time (a cache line, 16 `float`s), so how many *distinct* lines a gather
touches depends entirely on where the indices land — not on how many
elements are gathered.

If every index falls inside the same 16-float window, one line fetch
serves the whole gather. If the indices are spread across a large array
with no locality, almost every element needs its own line fetch — the
exact same number of gathered values, wildly different memory traffic.

## Task

`sol.hpp` gives you a line-tracking probe:

- `reset_lines()` clears the set of touched lines.
- `touch(p)` records the 64-byte line containing address `p`.
- `lines_touched()` returns how many distinct lines have been touched
  since the last reset.

Implement, in `solve.cpp`:

```cpp
void gather(const float* base, const int* idx, int n, float* result);
```

For `i` in $[0, n)$: `result[i] = base[idx[i]]`. Call `touch(&base[idx[i]])`
exactly once per element, in the same order you read it — reading
`base[idx[i]]` pulls in the cache line(s) covering that float.

The driver (`main.cpp`, fixed) builds a 64-byte-aligned `base` array of
2048 floats (`base[j] == j`) and runs your `gather` over two fixed,
deterministic 200-element index arrays:

- `idx_local[i] = i % 16` — every index lands in $[0, 16)$, one line's
  worth of floats.
- `idx_scattered[i] = (i * 97) \bmod 2048` — a prime stride spreads
  indices across the whole array.

## Example

$16$ floats is exactly one 64-byte line (`16 * 4 == 64`), so every
`idx_local` gather touches the same single line no matter how many of the
200 elements are read:

```
1468.000000 1
```

($1468$ = 13 copies each of $0..7$ plus 12 copies each of $8..15$, from how
`i % 16` distributes over 200 draws.)

`idx_scattered` visits residues spread across all $2048 / 16 = 128$ lines
in the array and, with 200 draws over a 97-stride, touches every one of
them:

```
199740.000000 128
ratio=128.000000
```

Same gather logic, same element count (200), 128x the distinct cache
lines — purely from where the indices point.

## What the gate checks

The grader compiles `main.cpp` + `solve.cpp` with `clang++ -O2 -std=c++20`,
runs it, extracts every printed number, and requires `max_abs_err <= 1e-6`
against the same driver linked with `ref.cpp`. The starter never reads
`base` and never calls `touch()`, so both sums print `0.000000` with `0`
lines each and `ratio=0.000000` — getting the gathered values right without
calling `touch(&base[idx[i]])` at the right address (or in the right
order) leaves the line counts wrong and still fails the gate.
