#include "sol.hpp"

// Stepping by exactly `sets` lines (S = line_bytes * sets) advances the
// line index by `sets` every step, so (line_index % sets) stays at 0 for
// every k -- the minimum stride that keeps all accesses in the same set.
long collision_stride(int line_bytes, int sets) {
    return static_cast<long>(line_bytes) * static_cast<long>(sets);
}
