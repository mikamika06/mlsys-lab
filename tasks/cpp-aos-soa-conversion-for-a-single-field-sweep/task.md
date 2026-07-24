## Context

When processing arrays of structs (AoS), sweeping a single field loads entire
cache lines that contain every field of the record — most of the data fetched
is unused. Converting to Struct of Arrays (SoA) places all values of one field
contiguously, so a single-field sweep fetches only what it needs.

The fixed record here is real C++, laid out by the actual compiler:

```cpp
struct Particle {
    float x, y, z;
    int id;
};
```

Four 4-byte members, naturally aligned, so `sizeof(Particle) == 16` with no
padding. A 64-byte cache line holds 4 `Particle` records but 16 lone `float`s.
For $N$ elements, sweeping field `x`:

$$\text{lines}_{\text{AoS}} = \left\lceil \frac{N \cdot \text{sizeof(Particle)}}{64} \right\rceil, \qquad
  \text{lines}_{\text{SoA}} = \left\lceil \frac{N \cdot \text{sizeof(float)}}{64} \right\rceil$$

The ratio $\text{lines}_{\text{SoA}} / \text{lines}_{\text{AoS}} = \text{sizeof(field)} / \text{sizeof(struct)} = 4/16 = 0.25$
is the bandwidth saved by SoA — measured here, not assumed.

## Task

The driver (`main.cpp`, fixed) allocates a 64-byte-aligned array of 1024
`Particle` (AoS) and a 64-byte-aligned array of 1024 `float` holding the same
`x` values (SoA), and gives you a cache-line probe declared in `sol.hpp`:

- `cache_reset()` clears the set of touched lines.
- `touch(p)` records the 64-byte line containing address `p`.
- `lines_touched()` returns how many distinct lines have been touched since
  the last reset.

Implement, in `solve.cpp`:

- `sum_field_aos(const Particle* arr, int n)` — sum `arr[i].x` over all `n`
  elements. Call `touch(&arr[i])` once per element: reading any field of
  `arr[i]` pulls in the whole record's cache line, so that's the address that
  represents the access.
- `sum_field_soa(const float* xs, int n)` — sum `xs[i]` over all `n`
  elements. Call `touch(&xs[i])` once per element.

## Example

With $N = 1024$: $\text{lines}_{\text{AoS}} = \lceil 1024 \cdot 16 / 64 \rceil = 256$,
$\text{lines}_{\text{SoA}} = \lceil 1024 \cdot 4 / 64 \rceil = 64$. Both sweeps
sum the same values ($0, 1, \dots, 1023$), so both sums must agree
(`sum = 523776`) — only the line counts differ:

```
523776.000000 256
523776.000000 64
ratio=0.250000
```

The starter never calls `touch()` and returns `0`, so it prints
`0.000000 0` twice and `ratio=0.000000`.

## What the gate checks

The grader compiles `main.cpp` + `solve.cpp` with `clang++ -O2 -std=c++20`,
runs it, extracts every printed number, and requires `max_abs_err <= 1e-6`
against the same driver linked with `ref.cpp`. Getting the sum right but
forgetting to call `touch()` (or touching the wrong address, e.g. per-field
instead of per-record in the AoS case) leaves the line counts wrong and fails
the gate — the lesson has to show up in the printed cache-line counts, not
just the sum.
