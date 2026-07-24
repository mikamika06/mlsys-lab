## Context

Array-of-Structs (AoS) stores each record contiguously (`x0,y0,z0,id0,
x1,y1,z1,id1,...`); Struct-of-Arrays (SoA) stores each *field* contiguously
(`x0,x1,...,y0,y1,...`). SoA is often better for SIMD/cache behavior when
a kernel only touches a few fields per record, but the conversion between
the two layouts has to be exact — it's pure data movement, no computation,
so there's no excuse for the round trip to change a single bit.

## Task

Implement

```cpp
void aos_to_soa(const Particle* aos, int n, float* xs, float* ys, float* zs, int* ids);
void soa_to_aos(const float* xs, const float* ys, const float* zs, const int* ids, int n,
                 Particle* aos);
```

where `Particle` (declared in `sol.hpp`) is `{ float x, y, z; int id; }`.
`aos_to_soa` scatters each field of `aos[i]` into `xs[i]`/`ys[i]`/`zs[i]`/
`ids[i]`. `soa_to_aos` is its exact inverse: gather `xs[i]`/`ys[i]`/
`zs[i]`/`ids[i]` back into `aos[i]`.

## Example

For `aos[3] = {1.5f, -2.0f, 0.25f, 42}`: after `aos_to_soa`,
`xs[3]==1.5f`, `ys[3]==-2.0f`, `zs[3]==0.25f`, `ids[3]==42`. Feeding those
back through `soa_to_aos` must reproduce `aos[3]` exactly.

## What the gate checks

`main.cpp` builds a deterministic array of 50 particles, round-trips it
through your `aos_to_soa` then `soa_to_aos`, and `memcmp`s every
reconstructed particle against the original — byte-for-byte, no tolerance.
It prints a line per particle plus the match count. The grader compiles
your `.cpp` with the real local `clang++`, runs it, and requires the
printed output to match the reference's exactly
($\mathrm{exact\_match}=1.0$). Mixing up which SoA array holds which field
(e.g. swapping `ys`/`zs`) still "round-trips" internally but doesn't match
the reference's field assignment, so it fails too.
