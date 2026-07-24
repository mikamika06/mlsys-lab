## Context

**AoSoA** (Array of Structures of Arrays) is a compromise between AoS (bad for SIMD/streaming: reading just one field forces the cache to pull in every other field too) and full SoA (great for streaming a single field, but scatters a *single particle's* fields across the whole array). AoSoA groups particles into fixed-size **tiles**; within a tile, each field is stored contiguously (SoA), and tiles are laid out one after another. The tile width `T` controls the trade-off: it changes exactly which byte addresses a given access pattern touches, and therefore how many real cache lines get pulled in.

## Task

Implement `generateAoSoATrace(tileWidth)` in `solve.cpp`, per the exact layout and address formula documented in `sol.hpp`: for `NUM_PARTICLES` particles with `NUM_FIELDS` fields each (x, y, z, mass -- field indices 0..3), compute the byte address of particle `i`'s field `f` under tile width `tileWidth`, and report every touched address for fields x, y, z (not mass) of every particle, in order, through `cacheTouch()`.

## Example

```cpp
// tileWidth = 8, NUM_FIELDS = 4, field bytes = 4
// particle i=10: tileIdx=1, withinTile=2, tileBytes=8*4*4=128
// field x (f=0): addr = 1*128 + 0*8*4 + 2*4 = 136
// field y (f=1): addr = 1*128 + 1*8*4 + 2*4 = 168
```

## What the gate checks

`main.cpp` defines a real, deterministic direct-mapped cache (64-byte lines, 32 lines -- never a real hardware counter, never wall-clock) behind the `cacheTouch()` hook, resets it, and runs `generateAoSoATrace` for tile widths `{1, 2, 4, 8, 16, 32, 64}` against a fixed 512-particle workload, printing the real access and miss counts the cache model observed for each. Your printed output is compared against `ref.cpp`, compiled and run the same way: `max_abs_err <= 1e-9`. Ignoring `tileWidth` (treating the layout as one flat SoA array regardless of tile size) produces the right address stream only for `tileWidth == NUM_PARTICLES`; every smaller tile width in the sweep gets the wrong addresses and therefore the wrong real miss count.
