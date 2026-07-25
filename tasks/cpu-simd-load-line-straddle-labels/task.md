## Context

The CPU fetches memory in fixed-size, aligned chunks called cache lines
(64 bytes on virtually every modern desktop/server core). A load that fits
entirely inside one line is a single fetch. A load whose byte range spans
*two* lines — because its starting address isn't aligned to the load's own
width — needs two fetches and, on many microarchitectures, an extra
penalty cycle or two just for the split itself. This matters most for wide
SIMD loads: a scalar 8-byte load can only straddle if it starts within the
last 7 bytes of a line, but a 64-byte AVX-512 `zmm` load straddles unless
its address is *exactly* line-aligned — any offset at all guarantees a
split.

## Task

Implement, in `solve.cpp`:

```cpp
bool straddles_line(uint64_t base_addr, int width_bytes, int line_bytes);
```

A load reads `[base_addr, base_addr + width_bytes)`. Compute
`offset = base_addr % line_bytes` (how far into its line the load starts)
and return `true` if `offset + width_bytes > line_bytes` — the load's last
byte falls past the end of the line it started in — and `false` otherwise.
A load that ends *exactly* on the boundary (`offset + width_bytes ==
line_bytes`) does **not** straddle: the boundary byte itself is one past
the last byte the load actually reads.

## Example

The driver (`main.cpp`, fixed) runs 8 loads against 64-byte lines, at a
fixed line-aligned base address, covering scalar (8-byte), SSE (16-byte),
AVX (32-byte) and AVX-512 (64-byte) widths:

```
scalar_aligned base=65536 width=8 straddle=0
zmm_full_line_aligned base=65536 width=64 straddle=0
xmm_ends_exactly_on_boundary base=65584 width=16 straddle=0
ymm_crosses_boundary base=65584 width=32 straddle=1
zmm_mid_line base=65568 width=64 straddle=1
scalar_worst_case base=65599 width=8 straddle=1
zmm_off_by_one base=65537 width=64 straddle=1
xmm_mid_line_fits base=65552 width=16 straddle=0
```

`xmm_ends_exactly_on_boundary` starts 48 bytes into its line and reads 16
more, landing exactly on byte 64 — the edge case that must read `false`.
`zmm_off_by_one` starts just 1 byte past a line boundary and reads a full
64-byte `zmm` register: even that single byte of misalignment is enough to
force a straddle, since a 64-byte load only fits within one 64-byte line
when it starts exactly on that line's boundary.

## What the gate checks

The grader compiles `main.cpp` + `solve.cpp` with `clang++ -O2 -std=c++20`,
runs it, and requires the entire printed output to match the reference
(`main.cpp` + `ref.cpp`) byte-for-byte (`exact_match == 1.0`). Using `>=`
instead of `>` in the comparison flips both loads that end exactly on a
boundary (`xmm_ends_exactly_on_boundary` and `zmm_full_line_aligned`) to
the wrong verdict; the starter always returns `false`, which is wrong on
the four straddling loads.
