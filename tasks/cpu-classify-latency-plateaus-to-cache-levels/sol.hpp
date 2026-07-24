#pragma once

// ============================================================================
// Four levels of the memory hierarchy, ordered by distance from the core.
// ============================================================================
enum class CacheLevel : int { L1 = 0, L2 = 1, L3 = 2, DRAM = 3 };

// ============================================================================
// LEARNER implements this in solve.cpp.
//
// classify_plateau: the driver (main.cpp, fixed) measures a *noisy* average
// latency (in simulated cycles) for a memory sweep whose working set is
// pinned entirely inside one level of the hierarchy. Each level has a fixed
// "true" latency roughly 3x its inner neighbour (see task.md) — but real
// measurements never land exactly on the plateau: the driver perturbs every
// sample by up to +-15% with a deterministic, seeded generator.
//
// Classify which level produced a given noisy latency sample. A hard
// equality check against the true latencies fails on every sample (the
// noise guarantees that); the fix is a nearest-boundary classifier using the
// geometric midpoint between each pair of neighbouring levels' true
// latencies as the decision threshold.
// ============================================================================
CacheLevel classify_plateau(double latency_cycles);
