#include "sol.hpp"

// BROKEN: no padding. stride stays 8, so all 8 threads' counters
// (addresses 0, 8, 16, ..., 56) land in the same 64-byte cache line --
// nearly every write in the driver's round-robin schedule is a
// false-sharing invalidation.
int counter_pad_bytes() {
    return 0;
}
