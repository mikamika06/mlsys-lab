#pragma once
#include <vector>

// ============================================================================
// LEARNER implements this in solve.cpp.
//
// dependency_chain_length: `depends_on[i]` is the index of the memory
// access that access `i` must WAIT for -- e.g. a pointer-chase step where
// `i`'s address is only known once access `depends_on[i]`'s value comes
// back -- or `-1` if access `i`'s address is known up front and it doesn't
// need to wait for anything. Guaranteed: whenever `depends_on[i] != -1`,
// `depends_on[i] < i`.
//
// Return the length of the LONGEST chain of accesses linked by
// `depends_on` -- the minimum number of SERIAL round trips a perfect,
// infinitely-wide out-of-order machine would still need, since accesses
// linked in a chain cannot overlap with each other no matter how much
// memory-level parallelism (MLP) the machine has. Independent accesses (no
// incoming chain) can all be in flight at once; that parallelism does not
// shorten anyone else's chain.
// ============================================================================
int dependency_chain_length(const std::vector<int>& depends_on);
