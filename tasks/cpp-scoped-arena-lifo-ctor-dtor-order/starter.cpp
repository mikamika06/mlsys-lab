#include "sol.hpp"
#include <new>       // placement new
#include <cstddef>   // std::size_t

// TODO: Build the scoped arena.
//   1. allocate one raw buffer of n * sizeof(Probe) bytes,
//   2. placement-new a Probe for each id (ids[0..n)) in FORWARD order,
//   3. destroy every Probe in strict LIFO (reverse-construction) order,
//   4. return the arena footprint in bytes: n * sizeof(Probe).
//
// The stub below compiles but constructs nothing, so no events are emitted
// and it fails the gate.
long run_scoped_arena(const int* ids, int n) {
    (void)ids;
    (void)n;
    return 0;
}
