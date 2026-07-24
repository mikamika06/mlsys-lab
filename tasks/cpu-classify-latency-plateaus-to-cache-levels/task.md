## Context

A pointer-chase benchmark that sweeps a working set through L1, L2, L3, and
DRAM produces a latency-per-access curve that looks like a staircase: flat
"plateaus" at each level's characteristic latency, with jumps between them
exactly where the working set stops fitting in one level and spills into the
next. Reading off *which* plateau a given latency belongs to is how you turn
a raw timing curve into a labeled cache-level ladder.

Real measurements are never perfectly flat — thermal noise, prefetcher
timing, and bus contention perturb every sample around its true plateau.
Here the four true per-level latencies (in simulated cycles) are fixed and
well separated, each roughly $3\times$ its inner neighbour:

$$L_1 = 4.0 \qquad L_2 = 12.0 \qquad L_3 = 36.0 \qquad \text{DRAM} = 140.0$$

and the driver perturbs every sample by up to $\pm 15\%$ before handing it to
your classifier — so a hard equality check against the true values fails on
every single sample.

## Task

Implement, in `solve.cpp`:

```cpp
CacheLevel classify_plateau(double latency_cycles);
```

`CacheLevel` is `L1 = 0, L2 = 1, L3 = 2, DRAM = 3` (declared in `sol.hpp`).
Given one noisy latency sample, decide which of the four true latencies it
was perturbed from.

The right decision rule is a nearest-boundary classifier using the
**geometric midpoint** between each pair of neighbouring true latencies as
the cutoff (geometric, not arithmetic, because the ladder is multiplicative —
each level is a *ratio* away from its neighbour, not a fixed offset):

$$
m_{12} = \sqrt{L_1 L_2} \approx 6.93 \qquad
m_{23} = \sqrt{L_2 L_3} \approx 20.78 \qquad
m_{34} = \sqrt{L_3 \cdot \text{DRAM}} \approx 70.99
$$

```
latency < m12         -> L1
m12 <= latency < m23   -> L2
m23 <= latency < m34   -> L3
latency >= m34          -> DRAM
```

With $\pm 15\%$ noise, the widest excursion from any true latency is still
comfortably inside its own bucket (e.g. DRAM's noise floor is $140 \times
0.85 = 119$, far above $m_{34} \approx 70.99$), so this rule classifies every
sample correctly.

## Example

The driver (`main.cpp`, fixed) generates 8 noisy samples per level, in level
order, using a seeded deterministic generator (no `rand()`, no timing), and
prints `<index> <latency> <level>` per sample. The first few lines look like:

```
0 3.740125 0
1 4.361883 0
...
8 12.977421 1
9 10.317202 1
...
24 32.552198 2
...
```

The starter always returns `CacheLevel::L1`, so from sample 8 onward its
printed labels diverge from the reference.

## What the gate checks

The grader compiles `main.cpp` + `solve.cpp` with `clang++ -O2 -std=c++20`,
runs it, and requires the entire printed output to match the reference
(`main.cpp` + `ref.cpp`) byte-for-byte (`exact_match == 1.0`). The latency
values themselves come only from the fixed driver, so they are identical for
every candidate — only a correct classification of *every* one of the 32
samples makes the output match. Getting most levels right but using the
wrong (arithmetic, not geometric) midpoint — or a single off-by-one boundary
— flips at least one label near a boundary and fails the gate.
