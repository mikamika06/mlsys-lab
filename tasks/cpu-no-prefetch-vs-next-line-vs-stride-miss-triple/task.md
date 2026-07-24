## Context

Hardware prefetchers try to guess future accesses from past ones without
any hint from software. Two classic policies:

- **Next-line**: after any miss on line `L`, blindly also fetch line
  `L+1`. Cheap, and works well when accesses are (close to) sequential —
  but if the real stride skips past the very next line, the prefetch is
  wasted.
- **Stride**: tracks the address delta between recent accesses; once the
  same delta repeats, it starts prefetching one delta ahead, correctly
  predicting patterns that skip by more than one line.

## Task

Implement

```cpp
void generate_and_run(long base, int stride_bytes, int n_steps, long* out);
```

Build the address trace `address_k = base + k * stride_bytes` for `k` in
`[0, n_steps)`, and feed each address, in order, through all three of
`touch_no_prefetch`, `touch_next_line`, `touch_stride` (declared in
`sol.hpp` — same address, same order, to all three, run against
independent private caches). Then write the three resulting miss counts
into `out[0]` (no-prefetch), `out[1]` (next-line), `out[2]` (stride).

## Example

With `stride_bytes = 128` (2 cache lines) and 64-byte lines: no-prefetch
misses on every one of the 40 steps (`40`). Next-line prefetch only ever
brings in the *adjacent* line after a miss, but the trace always skips
2 lines ahead, so every prefetch is wasted — also `40` misses, no
improvement at all. The stride prefetcher needs 2 accesses to learn the
128-byte delta, misses a 3rd time while confirming it, and from then on
correctly prefetches one stride ahead of every subsequent access: `3`
misses total.

## What the gate checks

`exact_match`: the driver prints the miss triple for one fixed 40-step,
128-byte-stride trace. Feeding the three functions different addresses
or a different order, or omitting a step, changes at least one of the
three counts; a starter that does nothing leaves the driver's `-1`
sentinels in place.
